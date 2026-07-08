#!/usr/bin/env python3
"""
Parallel Longest6 Evaluation Orchestrator.

Launches N CARLA server + Client container pairs in parallel, each handling
a subset of the 36 Longest6 routes. Aggregates metrics on completion.

Usage:
  python scripts/parallel_eval.py --model-path models/transfuser_official/model.pth --num-workers 4
  python scripts/parallel_eval.py --model-path models/transfuser_official/model.pth --num-workers 4 --dry-run
  python scripts/parallel_eval.py --model-path models/transfuser_official/model.pth --num-workers 4 --build
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel Longest6 Evaluation Orchestrator"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel CARLA+client pairs (default: 4)"
    )
    parser.add_argument(
        "--route-ids", type=str, default=None,
        help="Comma-separated route IDs to evaluate (default: all 0-35)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/metrics/longest6",
        help="Base metrics output directory (default: outputs/metrics/longest6)"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build Docker images before running"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--carla-timeout", type=int, default=300,
        help="Seconds to wait for each CARLA server to start (default: 300)"
    )
    parser.add_argument(
        "--worker-timeout", type=int, default=900,
        help="Per-worker timeout in seconds (default: 900)"
    )
    parser.add_argument(
        "--network", type=str, default="carla-longest6-net",
        help="Docker network name (default: carla-longest6-net)"
    )
    return parser.parse_args()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, dry_run=False, **kwargs):
    """Run a shell command, or print it if dry_run."""
    if dry_run:
        log(f"[DRY-RUN] {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    log(f"  Running: {' '.join(cmd[:4])}...")
    return subprocess.run(cmd, **kwargs)


def partition_routes(num_routes=36, num_workers=4, route_ids_str=None):
    """Partition routes into N balanced groups using round-robin."""
    if route_ids_str:
        ids = [r.strip() for r in route_ids_str.split(",")]
    else:
        ids = [str(i) for i in range(num_routes)]

    workers = [[] for _ in range(num_workers)]
    for i, rid in enumerate(ids):
        workers[i % num_workers].append(rid)
    return workers


def host_to_container_path(host_path, project_root):
    """Convert a host path to an in-container /app/... path."""
    host_path = os.path.abspath(host_path)
    project_root = os.path.abspath(project_root)
    if host_path.startswith(project_root):
        return "/app" + host_path[len(project_root):]
    return host_path


def build_images(project_root, dry_run=False):
    """Build CARLA and client Docker images."""
    log("Building Docker images...")

    # Build CARLA server image
    carla_compose_dir = os.path.join(project_root, "docker", "carla")
    log("  Building CARLA server image...")
    run_cmd(
        ["docker", "compose", "-f", os.path.join(carla_compose_dir, "docker-compose.yml"), "build"],
        dry_run=dry_run, check=True
    )

    # Build client image
    client_compose_dir = os.path.join(project_root, "docker", "client")
    log("  Building client image...")
    run_cmd(
        ["docker", "compose", "-f", os.path.join(client_compose_dir, "docker-compose.yml"), "build"],
        dry_run=dry_run, check=True
    )
    log("  Docker images built successfully.")


def ensure_network(network_name, dry_run=False):
    """Ensure the Docker bridge network exists."""
    log(f"Ensuring network '{network_name}' exists...")
    result = subprocess.run(
        ["docker", "network", "inspect", network_name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        run_cmd(
            ["docker", "network", "create", network_name, "--driver", "bridge"],
            dry_run=dry_run, check=True
        )


def launch_carla_worker(worker_id, network, dry_run=False):
    """Launch a CARLA server container. Returns the container name."""
    name = f"carla-server-{worker_id}"
    base_host_port = 2000 + worker_id * 10

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--network", network,
        "--rm",
        # Map host ports for optional debugging (unique per worker)
        "-p", f"{base_host_port}:2000",
        "-p", f"{base_host_port + 1}:2001",
        "-p", f"{base_host_port + 2}:2002",
        "carla-longest6:latest",
        "xvfb-run", "--auto-servernum",
        "--server-args=-screen 0 1024x768x24",
        "./CarlaUE4.sh", "-RenderOffScreen",
        "-quality-level=Low", "-carla-rpc-port=2000",
    ]

    result = run_cmd(cmd, dry_run=dry_run, capture_output=True, text=True, check=True)
    if not dry_run:
        container_id = result.stdout.strip()
        log(f"  CARLA worker {worker_id} started: {name} (ID: {container_id[:12]})")
    else:
        log(f"  CARLA worker {worker_id} would start: {name}")
    return name


def wait_for_carla_worker(worker_id, container_name, timeout=120, dry_run=False):
    """Wait for a CARLA server to accept TCP connections on port 2000."""
    if dry_run:
        return True

    log(f"  Waiting for CARLA worker {worker_id} ({container_name}:2000)...")
    for i in range(timeout + 1):
        result = subprocess.run(
            ["docker", "exec", container_name, "nc", "-z", "localhost", "2000"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log(f"  CARLA worker {worker_id} ready after {i}s")
            return True
        time.sleep(1)

    log(f"  ERROR: CARLA worker {worker_id} did not start within {timeout}s")
    return False


def launch_client_worker(worker_id, carla_host, routes, container_model_path,
                         output_dir, network, project_root, worker_timeout,
                         dry_run=False):
    """Launch an evaluation client container. Returns the container name."""
    name = f"transfuser-client-{worker_id}"
    route_ids = ",".join(routes)

    # Ensure worker output dir exists on host
    worker_out = os.path.join(output_dir, f"worker_{worker_id}")
    os.makedirs(worker_out, exist_ok=True)

    # Volume mounts (same as docker-compose.yml)
    volumes = [
        f"{os.path.join(project_root, 'transfuser')}:/app/transfuser",
        f"{os.path.join(project_root, 'config')}:/app/config",
        f"{os.path.join(project_root, 'scripts')}:/app/scripts",
        f"{os.path.join(project_root, 'models')}:/app/models",
        f"{os.path.join(project_root, 'outputs')}:/app/outputs",
    ]

    cmd = [
        "docker", "run", "--rm",
        "--name", name,
        "--network", network,
        "--runtime", "nvidia",
        "-e", f"CARLA_HOST={carla_host}",
        "-e", "CARLA_PORT=2000",
        "-e", f"MODEL_PATH={container_model_path}",
        "-e", f"ROUTE_IDS={route_ids}",
        "-e", f"OUTPUT_DIR=/app/outputs/metrics/longest6/parallel_{os.path.basename(output_dir)}/worker_{worker_id}",
        "-e", f"TRAFFIC_MANAGER_PORT={8000 + worker_id}",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
    ]
    # Add volume mounts
    for vol in volumes:
        cmd.extend(["-v", vol])

    cmd.append("transfuser-client:latest")

    if not dry_run:
        log(f"  Launching client {worker_id} (routes: {route_ids})...")
        # Run in background, capture output
        out_path = os.path.join(worker_out, "client_output.log")
        out_file = open(out_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True
        )
        return name, proc, out_file, out_path
    else:
        log(f"  [DRY-RUN] Would launch client {worker_id} (routes: {route_ids})")
        out_path = os.path.join(worker_out, "client_output.log")
        log(f"  [DRY-RUN]   Output: {out_path}")
        return name, None, None, out_path


def collect_worker_metrics(output_dir):
    """Collect all per-worker metrics JSON files."""
    combined = []
    output_path = Path(output_dir)

    for worker_dir in sorted(output_path.glob("worker_*/")):
        for json_file in sorted(worker_dir.glob("longest6_*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                combined.append(data)
            except (json.JSONDecodeError, IOError) as e:
                log(f"  WARNING: Could not read {json_file}: {e}")

    return combined


def aggregate_metrics(metrics_list, output_dir, start_time):
    """Aggregate per-worker metrics into a single combined metrics output."""
    if not metrics_list:
        log("  No metrics to aggregate.")
        return

    all_routes = []
    total_routes = 0
    total_completed = 0
    total_route_score = 0.0
    total_composed_score = 0.0
    total_penalty = 0.0
    model_name = metrics_list[0].get("model", "unknown")

    for m in metrics_list:
        all_routes.extend(m.get("per_route", []))
        total_routes += m.get("num_routes", 0)
        total_completed += m.get("num_completed", 0)
        total_route_score += m.get("route_completion_avg", 0) * m.get("num_routes", 0)
        total_composed_score += m.get("driving_score_avg", 0) * m.get("num_routes", 0)
        total_penalty += m.get("infraction_penalty_avg", 0) * m.get("num_routes", 0)

    duration = time.time() - start_time

    combined = {
        "model": model_name,
        "route": "Longest6 (Parallel)",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "num_workers": len(metrics_list),
        "num_routes": total_routes,
        "num_completed": total_completed,
        "route_completion_avg": round(total_route_score / max(total_routes, 1), 2),
        "driving_score_avg": round(total_composed_score / max(total_routes, 1), 2),
        "infraction_penalty_avg": round(total_penalty / max(total_routes, 1), 4),
        "per_worker_metrics": [m["duration_seconds"] for m in metrics_list],
        "per_route": all_routes,
    }

    # Write combined JSON
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"longest6_combined_{timestamp_str}_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    log(f"  Combined metrics: {json_path}")

    # Print summary
    print()
    print("=" * 55)
    print("  PARALLEL EVALUATION COMPLETE")
    print("=" * 55)
    print(f"  Model:        {model_name}")
    print(f"  Workers:      {len(metrics_list)}")
    print(f"  Routes:       {total_completed}/{total_routes} completed")
    print(f"  Avg RC:       {combined['route_completion_avg']:.1f}%")
    print(f"  Avg DS:       {combined['driving_score_avg']:.2f}")
    print(f"  Avg IP:       {combined['infraction_penalty_avg']:.4f}")
    print(f"  Wall time:    {duration:.0f}s ({duration/60:.1f}m)")
    print(f"  Total sim:    {sum(m.get('duration_seconds', 0) for m in metrics_list):.0f}s")
    print(f"  Speedup:      {sum(m.get('duration_seconds', 0) for m in metrics_list) / max(duration, 1):.1f}x")
    print("=" * 55)
    print()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))  # scripts/
    project_root = os.path.dirname(project_root)               # project root
    model_path = os.path.abspath(args.model_path)

    if not os.path.exists(model_path):
        log(f"ERROR: Model file not found: {model_path}")
        return 1

    # Partition routes
    workers = partition_routes(36, args.num_workers, args.route_ids)
    total_routes = sum(len(w) for w in workers)
    log(f"Partitioning {total_routes} routes across {args.num_workers} workers:")
    for i, routes in enumerate(workers):
        if routes:
            log(f"  Worker {i}: {len(routes)} routes [{','.join(routes)}]")

    # Build images if requested
    if args.build:
        build_images(project_root, args.dry_run)

    # Ensure network exists
    ensure_network(args.network, args.dry_run)

    # Clean up any stale containers from previous runs
    log("Cleaning up any stale containers...")
    for i in range(args.num_workers):
        subprocess.run(["docker", "rm", "-f", f"carla-server-{i}"],
                       capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", f"transfuser-client-{i}"],
                       capture_output=True, text=True)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parallel_out = os.path.join(
        os.path.abspath(args.output_dir), f"parallel_{timestamp}"
    )
    os.makedirs(parallel_out, exist_ok=True)

    start_time = time.time()

    # ===============================================================
    # Phase 1: Launch all CARLA servers in PARALLEL (shader compilation
    # can happen concurrently since it's CPU-bound and we have 176GB RAM).
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 1: Launching CARLA servers (parallel)")
    log(f"{'=' * 55}")

    carla_names = []
    for i in range(args.num_workers):
        if not workers[i]:
            log(f"  Worker {i}: no routes assigned, skipping")
            continue
        name = launch_carla_worker(i, args.network, args.dry_run)
        carla_names.append((i, name))

    if not carla_names:
        log("ERROR: No workers with routes assigned.")
        return 1

    # ===============================================================
    # Phase 2: Wait for all CARLA servers (parallel wait, check every 10s)
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 2: Waiting for all CARLA servers")
    log(f"{'=' * 55}")

    carla_ready = []
    remaining = list(carla_names)
    deadline = time.time() + args.carla_timeout
    while remaining and time.time() < deadline:
        still_waiting = []
        for i, name in remaining:
            result = subprocess.run(
                ["docker", "exec", name, "nc", "-z", "localhost", "2000"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                elapsed = int(time.time() - start_time)
                log(f"  CARLA worker {i} ready ({elapsed}s)")
                carla_ready.append((i, name, True))
            else:
                still_waiting.append((i, name))
        remaining = still_waiting
        if remaining:
            log(f"  Waiting for {len(remaining)} CARLA server(s): "
                f"{', '.join(n for _, n in remaining)}...")
            time.sleep(15)

    # Check which ones didn't make it
    for i, name in remaining:
        elapsed = int(time.time() - start_time)
        log(f"  ERROR: CARLA worker {i} not ready within {args.carla_timeout}s ({elapsed}s elapsed)")
        carla_ready.append((i, name, False))

    # If no CARLA servers are ready, abort
    if not any(ready for _, _, ready in carla_ready):
        log("  FATAL: No CARLA servers started successfully. Aborting.")
        cleanup_containers(carla_names, [], args.dry_run)
        return 1

    # Warn about failed servers but continue with what we have
    failed_count = sum(1 for _, _, ready in carla_ready if not ready)
    ready_count = sum(1 for _, _, ready in carla_ready if ready)
    if failed_count > 0:
        log(f"  WARNING: {failed_count} CARLA server(s) failed, continuing with {ready_count} worker(s)")

    # ===============================================================
    # Phase 3: Launch all client containers in parallel
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 3: Launching evaluation clients")
    log(f"{'=' * 55}")

    # Convert model path to container path
    container_model_path = host_to_container_path(model_path, project_root)
    log(f"  Model (host):      {model_path}")
    log(f"  Model (container): {container_model_path}")

    clients = []  # (worker_id, name, process, out_file, out_path)
    for i, name, ready in carla_ready:
        if not ready:
            mid = int(time.time() - start_time)
            log(f"  Skipping worker {i}: CARLA server not ready")
            continue
        carla_host = name  # Docker container name = hostname on bridge network
        c_name, proc, out_file, out_path = launch_client_worker(
            i, carla_host, workers[i], container_model_path,
            parallel_out, args.network, project_root,
            args.worker_timeout, args.dry_run
        )
        clients.append((i, c_name, proc, out_file, out_path))

    if args.dry_run:
        log(f"\n{'=' * 55}")
        log("DRY-RUN COMPLETE — no containers were actually launched.")
        log(f"{'=' * 55}")
        return 0

    # ===============================================================
    # Phase 4: Monitor clients with timeout per worker
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 4: Monitoring workers")
    log(f"{'=' * 55}")

    worker_results = {}
    for i, c_name, proc, out_file, out_path in clients:
        log(f"  Worker {i} ({c_name}): started, waiting up to {args.worker_timeout}s...")

    # Set up signal handler for graceful shutdown
    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    stop_requested = [False]

    def signal_handler(signum, frame):
        if stop_requested[0]:
            log("  Forcing immediate shutdown...")
            sys.exit(1)
        stop_requested[0] = True
        log("\n  Shutdown requested. Stopping workers... (Ctrl+C again to force)")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    deadline = time.time() + args.worker_timeout
    completed = 0
    failed = 0

    while clients and not stop_requested[0]:
        remaining = []
        for i, c_name, proc, out_file, out_path in clients:
            ret = proc.poll()
            if ret is not None:
                out_file.close()
                if ret == 0:
                    completed += 1
                    log(f"  Worker {i} completed successfully (exit=0)")
                    worker_results[i] = {"status": "completed", "exit_code": ret}
                else:
                    failed += 1
                    log(f"  Worker {i} FAILED (exit={ret}) — see {out_path}")
                    worker_results[i] = {"status": "failed", "exit_code": ret}
            else:
                if time.time() > deadline:
                    log(f"  Worker {i} TIMEOUT after {args.worker_timeout}s — killing...")
                    proc.kill()
                    out_file.close()
                    failed += 1
                    worker_results[i] = {"status": "timeout", "exit_code": -1}
                else:
                    remaining.append((i, c_name, proc, out_file, out_path))
        clients = remaining
        if clients and not stop_requested[0]:
            time.sleep(5)

    # Close any remaining file handles
    for _, _, _, out_file, _ in clients:
        try:
            out_file.close()
        except Exception:
            pass

    # ===============================================================
    # Phase 5: Clean up CARLA containers
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 5: Stopping CARLA servers")
    log(f"{'=' * 55}")

    cleanup_containers(carla_names, [], args.dry_run)

    # Restore signal handlers
    signal.signal(signal.SIGINT, orig_sigint)
    signal.signal(signal.SIGTERM, orig_sigterm)

    # ===============================================================
    # Phase 6: Collect and aggregate metrics
    # ===============================================================
    log(f"\n{'=' * 55}")
    log("PHASE 6: Aggregating results")
    log(f"{'=' * 55}")

    metrics_list = collect_worker_metrics(parallel_out)
    log(f"  Collected metrics from {len(metrics_list)} worker(s)")

    aggregate_metrics(metrics_list, parallel_out, start_time)

    log(f"  All outputs: {parallel_out}")

    if failed > 0:
        log(f"  WARNING: {failed}/{failed + completed} worker(s) had errors")
        return 1 if completed == 0 else 0

    return 0


def cleanup_containers(carla_names, client_names, dry_run=False):
    """Stop Docker containers."""
    for i, name in carla_names:
        log(f"  Stopping {name}...")
        subprocess.run(["docker", "stop", name], capture_output=True, text=True)

    for name in client_names:
        log(f"  Stopping {name}...")
        subprocess.run(["docker", "stop", name], capture_output=True, text=True)


if __name__ == "__main__":
    sys.exit(main())
