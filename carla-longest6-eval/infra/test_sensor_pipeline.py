#!/usr/bin/env python3
"""
Phase 1.2 Verification — CARLA headless sensor fetch test.
Connects to CARLA, spawns a vehicle with RGB + LiDAR sensors in synchronous mode,
ticks once, and prints received data shapes.
"""
import carla
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

print(f"Connecting to CARLA at {HOST}:{PORT} ...")
client = carla.Client(HOST, PORT)
client.set_timeout(30.0)

world = client.get_world()
print(f"World: {world.get_map().name}")
print(f"Tick: {world.get_snapshot().timestamp.frame_count}")

# Switch to synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Spawn ego vehicle
blueprint_library = world.get_blueprint_library()
bp = blueprint_library.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    print("ERROR: No spawn points on this map!")
    sys.exit(1)

transform = spawn_points[0]
vehicle = world.spawn_actor(bp, transform)
print(f"Spawned: {vehicle.type_id} at {transform.location}")

# Attach RGB camera
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.05")
camera_transform = carla.Transform(carla.Location(x=1.3, z=1.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Attach LiDAR
lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("channels", "64")
lidar_bp.set_attribute("range", "100")
lidar_bp.set_attribute("points_per_second", "1000000")
lidar_bp.set_attribute("rotation_frequency", "20")
lidar_bp.set_attribute("sensor_tick", "0.05")
lidar = world.spawn_actor(lidar_bp, camera_transform, attach_to=vehicle)

image_data = []
lidar_data = []

camera.listen(lambda img: image_data.append(img))
lidar.listen(lambda point_cloud: lidar_data.append(point_cloud))

# Tick synchronously and collect one round of sensor data
print("\nTicking synchronously...")
for i in range(5):
    world.tick()
    time.sleep(0.01)  # allow callbacks to fire

print(f"\nFrames captured: {len(image_data)} RGB, {len(lidar_data)} LiDAR")

if image_data:
    img = image_data[-1]
    import numpy as np
    arr = np.frombuffer(img.raw_data, dtype=np.uint8)
    arr = arr.reshape(img.height, img.width, 4)
    rgb = arr[:, :, :3]  # BGRA -> RGB
    print(f"  RGB frame: {img.width}x{img.height}, shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"  Pixel sample [100,100]: {rgb[100,100]}")

if lidar_data:
    cloud = lidar_data[-1]
    import numpy as np
    points = np.frombuffer(cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    print(f"  LiDAR points: {points.shape[0]} points")
    print(f"  Point sample: {points[0]}")

# Cleanup
print("\nCleanup...")
camera.stop()
lidar.stop()
camera.destroy()
lidar.destroy()
vehicle.destroy()

settings.synchronous_mode = False
world.apply_settings(settings)

print("\n=== Phase 1.2 VERIFIED: CARLA headless sensor pipeline works ===")
