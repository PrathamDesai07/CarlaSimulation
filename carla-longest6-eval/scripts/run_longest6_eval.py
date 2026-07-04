#!/usr/bin/env python3
"""
Longest6 Evaluation Script for CARLA 0.9.16 + TransFuser.

Evaluates autonomous driving performance on the fixed Longest6 route set (36 routes).
Wraps the CARLA Leaderboard evaluation framework in-process.

The agent (HybridAgent from submission_agent.py) requires a config DIRECTORY
containing args.txt (JSON) and one or more .pth model files. This script:
  1. Auto-detects the backbone type from checkpoint state dict keys
  2. Creates a temporary config directory with proper args.txt + model symlink
  3. Runs all Longest6 routes through LeaderboardEvaluator
  4. Extracts metrics from StatisticsManager and saves in our schema + leaderboard format

Real metrics produced (computed by the leaderboard evaluator from simulation):
  - Route Completion (%) per route and averaged
  - Driving Score (Route Completion × Infraction Penalty) per route and averaged
  - Infraction Penalty (multiplicative: 1.0 = no infractions) per route and averaged
  - Per-route infraction counts: collisions (pedestrian, vehicle, layout),
    red light, stop sign, outside route lanes, route deviation, timeout, vehicle blocked
"""

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
from datetime import datetime
from types import SimpleNamespace

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Longest6 Evaluation for CARLA + TransFuser"
    )
    parser.add_argument(
        "--carla-host", type=str, default="localhost",
        help="CARLA server hostname (default: localhost)"
    )
    parser.add_argument(
        "--carla-port", type=int, default=2000,
        help="CARLA server RPC port (default: 2000)"
    )
    parser.add_argument(
        "--traffic-manager-port", type=int, default=8000,
        help="Traffic manager port (default: 8000)"
    )
    parser.add_argument(
        "--traffic-manager-seed", type=int, default=0,
        help="Traffic manager random seed (default: 0)"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/longest6_config.yaml",
        help="Path to evaluation config YAML"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override metrics output directory"
    )
    parser.add_argument(
        "--route-id", type=str, default=None,
        help="Evaluate only a single route ID (e.g. '0') instead of all 36"
    )
    parser.add_argument(
        "--num-repetitions", type=int, default=1,
        help="Number of repetitions per route (default: 1)"
    )
    parser.add_argument(
        "--agent-timeout", type=float, default=300.0,
        help="Per-route agent timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--debug", type=int, default=0,
        help="Debug level: 0=minimal, 1=info, 2=verbose (default: 0)"
    )
    parser.add_argument(
        "--keep-config-dir", action="store_true",
        help="Preserve temporary config directory after run (for debugging)"
    )
    parser.add_argument(
        "--frames-dir", type=str, default=None,
        help="Directory to save video frames. If set, frames are saved and an MP4 is generated."
    )
    parser.add_argument(
        "--video-output", type=str, default=None,
        help="Output path for generated demo MP4 video (default: <metrics_dir>/demo_<timestamp>.mp4)"
    )
    return parser.parse_args()


def wait_for_carla(host, port, timeout=30):
    """Wait for CARLA server to accept TCP connections. Returns True if connected."""
    print(f"Waiting for CARLA at {host}:{port}...")
    for i in range(timeout + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"  CARLA ready after {i}s")
                return True
        except socket.error:
            pass
        if i < timeout:
            time.sleep(1)
    print(f"  ERROR: CARLA did not become ready after {timeout}s")
    return False


def load_config(path):
    """Load and return the evaluation configuration YAML."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def detect_backbone(checkpoint_path, config):
    """Auto-detect backbone and architecture from checkpoint state dict keys.

    Inspects the checkpoint's state dict keys to determine which model
    architecture to use. Falls back to config['model'] settings if detection
    is ambiguous.

    Returns:
        dict with keys: backbone, image_architecture, lidar_architecture,
                       use_velocity, use_target_point_image, use_point_pillars,
                       sync_batch_norm, n_layer, wp_only
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())

    # Build a key sample string for heuristics
    all_keys_str = " ".join(keys)

    # Determine backbone from checkpoint keys
    if "crossvit" in all_keys_str.lower():
        backbone = "crossvit_fusion"
    elif "transformer" in all_keys_str.lower():
        backbone = "transFuser"
    elif "gpt" in all_keys_str.lower():
        backbone = "transFuser"
    else:
        backbone = config.get("model", {}).get("backbone", "transFuser")
        print(f"  Backbone detection heuristic ambiguous, using config default: {backbone}")

    # Get architecture specs for this backbone
    model_specs = config.get("model_specs", {})
    if backbone in model_specs:
        img_arch = model_specs[backbone].get("image_architecture", "resnet34")
        lidar_arch = model_specs[backbone].get("lidar_architecture", "resnet18")
        use_vel = model_specs[backbone].get("use_velocity", True)
    else:
        img_arch = "resnet34"
        lidar_arch = "resnet18"
        use_vel = True
        print(f"  No model_specs entry for '{backbone}', using defaults")

    backbone_params = {
        "backbone": backbone,
        "image_architecture": img_arch,
        "lidar_architecture": lidar_arch,
        "use_velocity": use_vel,
        "use_target_point_image": True,
        "use_point_pillars": False,
        "sync_batch_norm": True,
        "n_layer": 4,
        "wp_only": 0,
    }

    print(f"  Detected backbone: {backbone}")
    print(f"    image_architecture: {img_arch}")
    print(f"    lidar_architecture: {lidar_arch}")
    print(f"    use_velocity: {use_vel}")

    return backbone_params


def create_agent_config_dir(model_path, backbone_params):
    """Create a temporary directory with args.txt + model symlink.

    HybridAgent.setup() expects a directory containing:
      - args.txt: JSON with model architecture parameters
      - One or more .pth files (it scans the directory)

    Returns path to the temp config directory.
    """
    config_dir = tempfile.mkdtemp(prefix="longest6_config_")

    # Write args.txt
    args_path = os.path.join(config_dir, "args.txt")
    with open(args_path, "w") as f:
        json.dump(backbone_params, f, indent=2)

    # Symlink the model checkpoint
    model_filename = os.path.basename(model_path)
    model_link = os.path.join(config_dir, model_filename)
    os.symlink(os.path.abspath(model_path), model_link)

    print(f"  Agent config dir: {config_dir}")
    return config_dir


def build_leaderboard_args(args, agent_config_dir, config):
    """Build a SimpleNamespace mimicking argparse for LeaderboardEvaluator.

    LeaderboardEvaluator.__init__ and .run() access these attributes.
    """
    paths = config.get("paths", {})
    routes_file = paths.get("routes", "transfuser/transfuser-2022/leaderboard/data/longest6/longest6.xml")
    scenarios_file = paths.get("scenarios", "transfuser/transfuser-2022/leaderboard/data/longest6/eval_scenarios.json")
    agent_script = paths.get("agent_script",
                             "transfuser/transfuser-2022/team_code_transfuser/submission_agent.py")
    checkpoint_path = paths.get("checkpoint_output",
                                f"{args.output_dir or 'outputs/metrics/longest6'}/leaderboard_checkpoint.json")

    # Ensure checkpoint dir exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    return SimpleNamespace(
        host=args.carla_host,
        port=str(args.carla_port),
        timeout=str(args.agent_timeout),
        trafficManagerPort=str(args.traffic_manager_port),
        trafficManagerSeed=str(args.traffic_manager_seed),
        debug=args.debug,
        record="",
        routes=routes_file,
        scenarios=scenarios_file,
        repetitions=args.num_repetitions,
        agent=agent_script,
        agent_config=agent_config_dir,
        track="SENSORS",
        resume=False,
        checkpoint=checkpoint_path,
    )


def extract_weather_name(weather):
    """Build a human-readable weather string from a carla.WeatherParameters."""
    if weather is None:
        return "Default"
    parts = []
    if weather.sun_altitude_angle < 0:
        parts.append("Night")
    elif weather.sun_altitude_angle < 20:
        parts.append("Dawn")
    elif weather.sun_altitude_angle > 70:
        parts.append("Noon")
    else:
        parts.append("Day")
    if weather.precipitation > 50:
        parts.append("Rain")
    elif weather.cloudiness > 50:
        parts.append("Cloudy")
    else:
        parts.append("Clear")
    return "_".join(parts)


def format_metrics(statistics_manager, route_indexer, start_time, args, model_name):
    """Extract metrics from StatisticsManager into our compact schema.

    StatisticsManager._registry_route_records contains RouteRecord objects
    with .scores, .infractions, .meta, .status, .route_id.
    """
    records = statistics_manager._registry_route_records
    model_spec = args.model_path

    total_routes = len(records)
    num_completed = sum(1 for r in records if r.status == "Completed")

    per_route = []
    total_route_score = 0.0
    total_composed_score = 0.0
    total_penalty = 0.0

    for r in records:
        # Extract route metadata — the RouteIndexer has configs with town/weather
        route_completion = r.scores.get("score_route", 0.0)
        infraction_penalty = r.scores.get("score_penalty", 1.0)
        driving_score = r.scores.get("score_composed", 0.0)

        total_route_score += route_completion
        total_composed_score += driving_score
        total_penalty += infraction_penalty

        # Try to extract town and weather from the route config
        town = "unknown"
        weather_name = "unknown"
        for key, cfg in route_indexer._configs_dict.items():
            if hasattr(cfg, 'name') and cfg.name == r.route_id:
                if hasattr(cfg, 'town'):
                    town = cfg.town
                if hasattr(cfg, 'weather'):
                    weather_name = extract_weather_name(cfg.weather)
                break
        # Fallback: parse route_id which looks like "RouteScenario_0"
        if r.route_id and r.route_id.startswith("RouteScenario_"):
            rid = r.route_id.split("_")[-1]
            if town == "unknown":
                town = f"Town{int(rid) % 10 + 1}"

        route_entry = {
            "route_id": r.route_id,
            "town": town,
            "weather": weather_name,
            "status": "Completed" if r.status == "Completed" else "Failed",
            "route_completion": round(route_completion, 2),
            "infraction_penalty": round(infraction_penalty, 4),
            "driving_score": round(driving_score, 2),
            "infractions": {
                "collisions_pedestrian": len(r.infractions.get("collisions_pedestrian", [])),
                "collisions_vehicle": len(r.infractions.get("collisions_vehicle", [])),
                "collisions_layout": len(r.infractions.get("collisions_layout", [])),
                "red_light": len(r.infractions.get("red_light", [])),
                "stop_infraction": len(r.infractions.get("stop_infraction", [])),
                "outside_route_lanes": len(r.infractions.get("outside_route_lanes", [])),
                "route_dev": len(r.infractions.get("route_dev", [])),
                "route_timeout": len(r.infractions.get("route_timeout", [])),
                "vehicle_blocked": len(r.infractions.get("vehicle_blocked", [])),
            },
            "route_length_m": round(r.meta.get("route_length", 0.0), 1),
            "duration_game": round(r.meta.get("duration_game", 0.0), 1),
            "duration_system": round(r.meta.get("duration_system", 0.0), 1),
        }
        per_route.append(route_entry)

    duration = time.time() - start_time if start_time else 0.0

    metrics = {
        "model": model_name,
        "route": "Longest6",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "num_routes": total_routes,
        "num_completed": num_completed,
        "route_completion_avg": round(total_route_score / max(total_routes, 1), 2),
        "driving_score_avg": round(total_composed_score / max(total_routes, 1), 2),
        "infraction_penalty_avg": round(total_penalty / max(total_routes, 1), 4),
        "per_route": per_route,
    }

    return metrics


def save_metrics(metrics, output_dir, model_name):
    """Save metrics in JSON format and a human-readable text summary.

    Returns dict of files written.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compact JSON
    json_path = os.path.join(output_dir, f"longest6_{timestamp_str}_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON: {json_path}")

    # Human-readable summary
    txt_path = os.path.join(output_dir, f"longest6_{timestamp_str}_{model_name}.txt")
    with open(txt_path, "w") as f:
        f.write("Longest6 Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model:     {model_name}\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Duration:  {metrics['duration_seconds']}s\n\n")

        f.write("Summary\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Routes evaluated:     {metrics['num_routes']}\n")
        f.write(f"  Routes completed:     {metrics['num_completed']}\n")
        f.write(f"  Route Completion Avg: {metrics['route_completion_avg']:.2f}%\n")
        f.write(f"  Driving Score Avg:    {metrics['driving_score_avg']:.2f}\n")
        f.write(f"  Infraction Penalty:   {metrics['infraction_penalty_avg']:.4f}\n\n")

        f.write("Per-Route Breakdown\n")
        f.write("-" * 30 + "\n")
        for r in metrics["per_route"]:
            f.write(f"  Route {r['route_id']} ({r['town']}, {r['weather']}):\n")
            f.write(f"    Status:              {r['status']}\n")
            f.write(f"    Route Completion:    {r['route_completion']:.1f}%\n")
            f.write(f"    Driving Score:       {r['driving_score']:.2f}\n")
            f.write(f"    Infraction Penalty:  {r['infraction_penalty']:.4f}\n")
            inf = r["infractions"]
            total_inf = sum(inf.values())
            if total_inf > 0:
                f.write(f"    Infractions ({total_inf} total):\n")
                for k, v in inf.items():
                    if v > 0:
                        f.write(f"      {k}: {v}\n")
            f.write(f"    Route Length:        {r['route_length_m']:.1f}m\n")
            f.write(f"    Duration:            {r['duration_game']:.1f}s\n\n")

        f.write("=" * 50 + "\n")

    print(f"  Summary TXT: {txt_path}")

    return {
        "json_path": json_path,
        "txt_path": txt_path,
    }


def run_evaluation(args, config):
    """Run the full Longest6 evaluation using the in-process leaderboard framework.

    Sets up PYTHONPATH for leaderboard imports, detects the model backbone,
    creates the agent config dir, instantiates LeaderboardEvaluator, runs
    all routes, then extracts and saves metrics.

    Returns 0 on success, 1 on failure.
    """
    model_path = os.path.abspath(args.model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.abspath(args.output_dir or config.get("metrics", {}).get(
        "output_dir", "outputs/metrics/longest6"
    ))

    start_time = time.time()

    # Step 1: Wait for CARLA server
    print("\n[1/6] Waiting for CARLA server...")
    if not wait_for_carla(args.carla_host, args.carla_port):
        print("ERROR: CARLA server is not running. Start it with:")
        print("  docker-compose -f docker/carla/docker-compose.yml up -d")
        print("  # or: bash scripts/run_carla_docker.sh up")
        return 1

    # Step 2: Detect backbone from checkpoint
    print("\n[2/6] Detecting model backbone...")
    try:
        backbone_params = detect_backbone(model_path, config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return 1

    # Step 3: Create temp agent config dir
    print("\n[3/6] Creating agent configuration...")
    config_dir = create_agent_config_dir(model_path, backbone_params)

    try:
        # Step 3b: Set up frame capture directory
        metrics_cfg = config.get("metrics", {})
        frame_dir = os.path.abspath(args.frames_dir or metrics_cfg.get("frame_dir", "outputs/frames/longest6"))
        os.makedirs(frame_dir, exist_ok=True)
        os.environ['FRAME_PATH'] = frame_dir
        print(f"\n  Frames will be saved to: {frame_dir}")

        # Step 4: Set up PYTHONPATH for leaderboard framework imports
        print("\n[4/6] Setting up leaderboard framework...")
        project_root = os.path.dirname(os.path.abspath(__file__))  # scripts/
        project_root = os.path.dirname(project_root)               # carla-longest6-eval/
        transfuser_dir = os.path.join(project_root, "transfuser", "transfuser-2022")
        print(f"  Project root: {project_root}")
        # The leaderboard Python package is at
        # transfuser-2022/leaderboard/leaderboard/ — add its parent so
        # `from leaderboard.scenarios...` and similar internal imports work.
        sys.path.insert(0, os.path.join(transfuser_dir, "team_code_transfuser"))
        sys.path.insert(0, os.path.join(transfuser_dir, "leaderboard"))
        # scenario_runner is imported by leaderboard evaluator
        sys.path.insert(0, os.path.join(transfuser_dir, "scenario_runner"))
        # CARLA PythonAPI for agents module (from CARLA server container)
        sys.path.insert(0, os.path.join(transfuser_dir, "PythonAPI", "carla"))
        sys.path.insert(0, os.path.join(transfuser_dir, "PythonAPI"))

        # Step 4b: import leaderboard evaluator
        from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
        from leaderboard.utils.statistics_manager import StatisticsManager
        from leaderboard.utils.route_indexer import RouteIndexer

        # Build leaderboard args namespace
        lb_args = build_leaderboard_args(args, config_dir, config)

        # Override routes file if single-route mode
        if args.route_id is not None:
            routes_file = lb_args.routes
            # Use single_route parameter in a custom way — override the XML path
            # to only contain the single route by creating a filtered temp file
            import xml.etree.ElementTree as ET
            tree = ET.parse(routes_file)
            root = tree.getroot()
            for route_elem in list(root):
                if route_elem.tag == "route" and route_elem.attrib.get("id") != args.route_id:
                    root.remove(route_elem)
            filtered_path = routes_file.rsplit(".", 1)[0] + f"_filtered_{args.route_id}.xml"
            tree.write(filtered_path)
            lb_args.routes = filtered_path
            print(f"  Filtered to single route ID: {args.route_id}")

        # Step 5: Run evaluation
        print("\n[5/6] Running Longest6 evaluation...")
        print(f"  Model: {model_path}")
        print(f"  Routes: {lb_args.routes}")
        print(f"  Scenarios: {lb_args.scenarios}")
        print(f"  Checkpoint: {lb_args.checkpoint}")
        print(f"  Repetitions: {lb_args.repetitions}")
        print()

        statistics_manager = StatisticsManager()
        evaluator = LeaderboardEvaluator(lb_args, statistics_manager)
        evaluator.run(lb_args)

        # Step 6: Format and save metrics
        print("\n[6/6] Saving results...")
        metrics = format_metrics(statistics_manager, evaluator.statistics_manager, start_time, args, model_name)

        files = save_metrics(metrics, output_dir, model_name)

        print(f"\n=== Evaluation Complete ===")
        print(f"  Routes completed: {metrics['num_completed']}/{metrics['num_routes']}")
        print(f"  Avg Route Completion: {metrics['route_completion_avg']:.1f}%")
        print(f"  Avg Driving Score:    {metrics['driving_score_avg']:.2f}")
        print(f"  Avg Infraction Penalty: {metrics['infraction_penalty_avg']:.4f}")
        print(f"  Duration: {metrics['duration_seconds']:.0f}s")

        # Generate demo video from captured frames
        video_output = args.video_output or os.path.join(metrics_cfg.get("video_dir", "outputs/videos/longest6"),
                                                          f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        video_dir = os.path.dirname(video_output)
        os.makedirs(video_dir, exist_ok=True)
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        if frame_files:
            print(f"\n  Generating demo video from {len(frame_files)} frames...")
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "10",
                "-pattern_type", "glob",
                "-i", os.path.join(frame_dir, "*.png"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                video_output
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Video saved to: {video_output}")
            else:
                print(f"  ffmpeg failed (return code {result.returncode}): {result.stderr[:200]}")
        else:
            print(f"\n  No frames found in {frame_dir} — skipping video generation.")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: File not found — {e}")
        print("Make sure the leaderboard framework files are present in transfuser/transfuser-2022/")
        traceback.print_exc()
        return 1
    except ImportError as e:
        print(f"\nERROR: Import failed — {e}")
        print("Check that CARLA client library and all Python dependencies are installed.")
        traceback.print_exc()
        return 1
    except SystemExit as e:
        # LeaderboardEvaluator may call sys.exit() on simulation crashes
        # We catch this to still save metrics for completed routes
        print(f"\n  Leaderboard exited with code {e.code}")
        return 0 if e.code == 0 else 1
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Clean up temp config dir (unless --keep-config-dir)
        if not args.keep_config_dir and 'config_dir' in dir():
            shutil.rmtree(config_dir, ignore_errors=True)
            print("  Cleaned up temporary config directory.")

        # Clean up filtered route file if created
        if args.route_id is not None:
            filtered = lb_args.routes if 'lb_args' in dir() else ""
            if filtered and "_filtered_" in filtered:
                try:
                    os.remove(filtered)
                except OSError:
                    pass


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    return run_evaluation(args, config)


if __name__ == "__main__":
    sys.exit(main())
