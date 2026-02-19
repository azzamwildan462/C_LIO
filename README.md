# DLIO-SCLC: Direct LiDAR-Inertial Odometry with Scan Context Loop Closure

An extended version of [Direct LiDAR-Inertial Odometry (DLIO)](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) with **Graph SLAM loop closure**, **Scan Context relocalization**, and **prior map localization** support.

## What's New

This fork adds three major capabilities on top of the original DLIO:

| Feature | Description |
|---------|-------------|
| **Graph SLAM** | Pose graph optimization with automatic loop closure detection using GICP + g2o |
| **Scan Context Relocalization** | Automatic initial pose estimation when loading a prior map, using Scan Context descriptors + GICP refinement |
| **Keyframe Database (KFDB)** | Persistent storage of real keyframe Scan Context descriptors for accurate relocalization |
| **Prior Map Localization** | Load a previously built map and localize against it |
| **Continuous Localization** | Periodic background GICP against global prior map with `map→odom` TF correction (like RTAB-Map) |
| **Occupancy Grid Map** | Probabilistic 2D occupancy grid from LiDAR scans using Bresenham ray tracing + log-odds Bayesian filter |
| **Composable Nodes** | All nodes run in a single process with intra-process communication for lower latency |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   component_container_mt                      │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │   OdomNode   │──>│   MapNode    │   │  GraphSlamNode  │  │
│  │              │   │              │   │                 │  │
│  │ - IMU fusion │   │ - Map accum  │   │ - Loop detect   │  │
│  │ - GICP align │   │ - Auto-save  │   │ - Pose graph    │  │
│  │ - Relocalize │   │ - PCD I/O    │   │ - g2o optimize  │  │
│  │ - KFDB save  │   │              │   │                 │  │
│  └──────────────┘   └──────────────┘   └─────────────────┘  │
│        IPC              pub/sub              IPC             │
└──────────────────────────────────────────────────────────────┘
```

All three nodes run as composable components in a single multi-threaded container, with intra-process communication (IPC) enabled for OdomNode and GraphSlamNode.

## Code Structure

The `OdomNode` class is split across multiple source files for maintainability. All files implement methods of the same `dlio::OdomNode` class via separate translation units.

```
include/dlio/
├── odom.h                  # OdomNode class declaration
├── map.h                   # MapNode class declaration
├── graph_slam.h            # GraphSlamNode class declaration
├── scan_context.h          # Shared Scan Context utilities (header-only)
├── kfdb_io.h               # Keyframe Database I/O interface
├── occupancy_grid.h        # Probabilistic occupancy grid generator
├── utils.h                 # Common types and utilities
└── dlio.h                  # Package-wide defines

src/dlio/
├── odom.cc                 # Constructor, getParams(), start()
├── odom_callbacks.cc       # callbackPointCloud, callbackImu, deskewing, preprocessing
├── odom_registration.cc    # GICP/NDT alignment, IMU integration, state propagation
├── odom_keyframes.cc       # Keyframe management, submap building, metrics
├── odom_relocalization.cc  # Prior map loading, SC database, relocalization, KFDB save/load
├── odom_services.cc        # ROS services, publishing, continuous localization, debug
├── occupancy_grid.cc       # Probabilistic occupancy grid map generator
├── kfdb_io.cc              # KFDB binary file read/write (standalone module)
├── map.cc                  # MapNode implementation
├── map_node.cc             # MapNode component registration
├── graph_slam.cc           # GraphSlamNode implementation
└── graph_slam_node.cc      # GraphSlamNode component registration
```

### Shared Modules

- **`scan_context.h`** — Header-only library (`namespace dlio::sc`) containing Scan Context descriptor computation, ring key extraction, distance calculation, and gravity-aligned scan preparation. Used by both `OdomNode` and `GraphSlamNode`.
- **`kfdb_io.h` / `kfdb_io.cc`** — Standalone KFDB binary file I/O (`namespace dlio::kfdb`). Handles reading and writing the `.kfdb` keyframe database files independently of the node class.

## Dependencies

- Ubuntu 22.04
- ROS 2 Humble
- C++ 17
- Point Cloud Library >= 1.10.0
- Eigen >= 3.3.7
- g2o (via `ros-humble-libg2o`)
- [ndt_omp](https://github.com/koide3/ndt_omp) (optional NDT registration backend)
- OpenMP >= 4.5

```bash
sudo apt install libomp-dev libpcl-dev libeigen3-dev ros-humble-libg2o ros-humble-pcl-ros
```

## Build

```bash
cd <your_ws>
colcon build --packages-select direct_lidar_inertial_odometry
source install/setup.bash
```

## Usage

### Mapping

Build a map from a rosbag:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  map_mode:=mapping \
  map_path:=/path/to/my_map.pcd \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/ouster/imu
```

This produces two files:
- `my_map.pcd` — the point cloud map
- `my_map.kfdb` — keyframe database for relocalization

### Localization (known initial pose)

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  map_mode:=localization \
  map_path:=/path/to/my_map.pcd \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/ouster/imu
```

### Localization with Relocalization (unknown initial pose)

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  map_mode:=localization \
  map_path:=/path/to/my_map.pcd \
  relocalize:=true \
  pointcloud_topic:=/ouster/points \
  imu_topic:=/ouster/imu
```

The system will automatically determine the initial pose using Scan Context matching + GICP refinement before starting odometry tracking.

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `map_mode` | `localization` | `mapping` or `localization` |
| `map_path` | `""` | Path to PCD map file (empty = `$HOME/.ros/dlio_map.pcd`) |
| `relocalize` | `true` | Enable Scan Context relocalization |
| `use_corrected` | `true` | Load graph-optimized `_corrected` map/KFDB files in localization mode |
| `registration_method` | `gicp` | Point cloud registration: `gicp` or `ndt` |
| `pointcloud_topic` | `points_raw` | Input point cloud topic |
| `imu_topic` | `imu_raw` | Input IMU topic |
| `rviz` | `false` | Launch RViz |

## Runtime Services

All services are available under the `/dlio/odom_node/` namespace for runtime control without restarting the system.

### Get State

Query current mode, pose, orientation, and statistics.

```bash
ros2 service call /dlio/odom_node/get_state direct_lidar_inertial_odometry/srv/GetState
```

Response includes: `mode`, `relocalized`, `x`, `y`, `z`, `roll_deg`, `pitch_deg`, `yaw_deg`, `length_traversed`, `num_keyframes`.

### Set Mode

Switch between mapping and localization at runtime. When switching from mapping to localization, the map PCD and KFDB are auto-saved.

```bash
# Switch to mapping
ros2 service call /dlio/odom_node/set_mode direct_lidar_inertial_odometry/srv/SetMode "{mode: 'mapping'}"

# Switch to localization with a specific map
ros2 service call /dlio/odom_node/set_mode direct_lidar_inertial_odometry/srv/SetMode \
  "{mode: 'localization', map_path: '/path/to/map.pcd'}"
```

### Set Pose

Manually set the robot pose (keeps current roll/pitch from IMU, overrides yaw).

```bash
ros2 service call /dlio/odom_node/set_pose direct_lidar_inertial_odometry/srv/SetPose \
  "{x: 1.0, y: 2.0, z: 0.0, yaw_deg: 90.0}"
```

### Relocalize

Trigger SC+GICP relocalization. Blocks until relocalization succeeds or times out (30s). Returns the resulting pose and GICP fitness score.

```bash
ros2 service call /dlio/odom_node/relocalize direct_lidar_inertial_odometry/srv/Relocalize
```

### New Map

Clear all keyframes, map data in memory, and delete the map files (`.pcd` + `.kfdb`), then switch to mapping mode.

```bash
ros2 service call /dlio/odom_node/new_map direct_lidar_inertial_odometry/srv/NewMap
```

### New Map with Zero Pose

Same as New Map, but also resets the robot pose to origin `[0, 0, 0]` with yaw = 0.

```bash
ros2 service call /dlio/odom_node/new_map_w_zero direct_lidar_inertial_odometry/srv/NewMapWZero
```

### Save PCD (MapNode)

Manually trigger a map save via the MapNode.

```bash
ros2 service call /dlio/map_node/save_pcd direct_lidar_inertial_odometry/srv/SavePCD \
  "{leaf_size: 0.25, save_path: '/path/to/output'}"
```

### Service Summary

| Service | Topic | Description |
|---------|-------|-------------|
| `GetState` | `/dlio/odom_node/get_state` | Query current state and pose |
| `SetMode` | `/dlio/odom_node/set_mode` | Switch mapping/localization mode |
| `SetPose` | `/dlio/odom_node/set_pose` | Manually set robot pose |
| `Relocalize` | `/dlio/odom_node/relocalize` | Trigger SC+GICP relocalization |
| `NewMap` | `/dlio/odom_node/new_map` | Clear everything, start fresh mapping |
| `NewMapWZero` | `/dlio/odom_node/new_map_w_zero` | Clear everything + reset pose to zero |
| `SavePCD` | `/dlio/map_node/save_pcd` | Save current map to PCD file |

## Features in Detail

### Graph SLAM with Loop Closure

A dedicated `GraphSlamNode` runs alongside odometry and mapping:

1. Receives keyframes (pose + cloud) from the odometry node via `KeyframeStamped` messages
2. Detects loop closure candidates based on spatial proximity and temporal gap
3. Validates candidates using GICP alignment against a local submap
4. Optimizes the full pose graph using g2o (SE3 vertices + edges)
5. Publishes corrected trajectory, keyframe poses, and corrected map

Configuration in `cfg/graph_slam.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loop_closure/range` | `10.0` | Max distance (m) for loop candidates |
| `loop_closure/min_keyframe_gap` | `15` | Min keyframe index gap |
| `loop_closure/fitness_score_threshold` | `0.3` | GICP fitness threshold for acceptance |
| `pose_graph/optimization_iterations` | `20` | g2o Levenberg-Marquardt iterations |
| `pose_graph/num_adjacent_constraints` | `5` | Sequential odometry edges per keyframe |

### Scan Context Relocalization

When `relocalize:=true`, the system determines the initial pose automatically:

1. Computes a **Scan Context descriptor** (20 rings x 60 sectors) from the incoming LiDAR scan, rotated to gravity-aligned frame
2. Matches against the **Keyframe Database (KFDB)** loaded from the `.kfdb` file
3. Takes the **top-50 candidates** and tries **6 yaw hypotheses** per candidate (0°, 60°, 120°, 180°, 240°, 300°)
4. For each hypothesis, extracts a **50m-radius local map** and refines with **GICP alignment**
5. Accepts the best result if GICP fitness < 0.5, early-exits if < 0.3

Configuration in `cfg/params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map/relocalize/sc_max_range` | `40.0` | Max LiDAR range for SC descriptor (m) |
| `map/relocalize/sc_num_candidates` | `50` | Top-K SC candidates to try |
| `map/relocalize/sc_max_attempts` | `10` | Max scans before falling back to initial pose |

### Keyframe Database (KFDB)

During mapping, each keyframe's Scan Context descriptor is computed from the **raw sensor-frame scan** (rotated to gravity-aligned frame using IMU calibration data) and stored in a compact binary file (`.kfdb`, ~5 KB per keyframe).

The KFDB v2 format stores:
- **Header**: magic, version, SC dimensions (NR/NS), `sc_max_range`, gravity quaternion
- **Per keyframe**: SC descriptor, ring key, position, orientation

This ensures relocalization uses **real viewpoint-dependent descriptors** rather than synthetic ones generated from the dense map, dramatically improving matching accuracy.

The KFDB is auto-saved every 20 keyframes and on shutdown.

### Prior Map Loading

Both modes support loading a prior map from PCD:

- **Mapping mode**: Loads existing map as a starting point, continues accumulating
- **Localization mode**: Loads map, builds KdTree + spatial chunks, tracks against it

The prior map is spatially chunked (20m grid cells), with each chunk contributing a virtual keyframe for submap building.

### Continuous Localization

When enabled (`map/continuous_localize: true`), a background timer periodically runs GICP alignment of the latest scan against the **global prior map** to compute a `map→odom` TF correction. This follows the standard ROS 2 localization architecture (like RTAB-Map, AMCL):

```
TF tree:  map → odom → base_link → {imu, lidar}
              ↑           ↑
     continuous GICP    odometry (smooth)
```

- **`odom→base_link`**: Published by DLIO odometry (smooth, continuous)
- **`map→odom`**: Published by continuous localization (periodic correction for global drift)

External nodes that need the globally-corrected pose should look up `map→base_link` through the TF tree.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map/continuous_localize` | `true` | Enable periodic global GICP correction |
| `map/continuous_localize/interval` | `2.0` | Time between corrections (seconds) |
| `map/continuous_localize/fitness_threshold` | `0.5` | Max GICP fitness score to accept correction |
| `frames/map` | `map` | Map frame name for TF |

### Occupancy Grid Map

When enabled (`occupancy_grid/enabled: true`), a probabilistic 2D occupancy grid is generated from each LiDAR scan and published as `nav_msgs/OccupancyGrid`. The algorithm is inspired by [Autoware's pointcloud-based occupancy grid map](https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_probabilistic_occupancy_grid_map/pointcloud-based-occupancy-grid-map/).

**Algorithm** (3 steps per scan):

1. **Height-based classification**: Points are split into ground and obstacle based on height relative to the sensor origin
2. **Polar binning + Bresenham ray tracing**: Points are binned by angle (720 bins, 0.5 deg). For each bin, a ray is traced from the sensor origin marking cells as FREE, with obstacle endpoints marked OCCUPIED
3. **Log-odds Bayesian update**: Each cell maintains a persistent log-odds value updated additively with clamping to prevent overconfidence

The grid is an **ego-centric rolling window** (default 100m x 100m at 0.2m resolution) that shifts as the robot moves. Runs in a timer callback at configurable rate (default 5Hz) — independent of the LiDAR pipeline with zero impact on odometry performance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `occupancy_grid/enabled` | `false` | Master enable/disable |
| `occupancy_grid/grid_size_x` | `100.0` | Grid width in meters |
| `occupancy_grid/grid_size_y` | `100.0` | Grid height in meters |
| `occupancy_grid/resolution` | `0.2` | Cell size in meters |
| `occupancy_grid/ground_threshold` | `-0.3` | Height below sensor origin treated as ground (m) |
| `occupancy_grid/obstacle_min_height` | `0.1` | Min height above sensor for obstacle (m) |
| `occupancy_grid/obstacle_max_height` | `3.0` | Max height for obstacle (m) |
| `occupancy_grid/p_occupied` | `0.7` | Inverse sensor model: P(occ \| obstacle) |
| `occupancy_grid/p_free` | `0.3` | Inverse sensor model: P(occ \| free) |
| `occupancy_grid/lo_clamped_min` | `-4.0` | Min log-odds (probability ~0.018) |
| `occupancy_grid/lo_clamped_max` | `4.0` | Max log-odds (probability ~0.982) |
| `occupancy_grid/decay_rate` | `0.0` | Time decay rate (0 = disabled) |
| `occupancy_grid/obstacle_margin` | `0.3` | Extra margin around obstacles (m) |
| `occupancy_grid/update_rate` | `5.0` | Update + publish rate (Hz) |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `dlio/odom_node/odom` | `nav_msgs/Odometry` | Odometry estimate |
| `dlio/odom_node/pose` | `geometry_msgs/PoseStamped` | Current pose |
| `dlio/odom_node/path` | `nav_msgs/Path` | Trajectory path |
| `dlio/odom_node/pointcloud/deskewed` | `sensor_msgs/PointCloud2` | Deskewed scan in world frame |
| `dlio/odom_node/pointcloud/keyframe` | `sensor_msgs/PointCloud2` | Keyframe cloud |
| `dlio/odom_node/keyframes` | `geometry_msgs/PoseArray` | All keyframe poses |
| `dlio/odom_node/occupancy_grid` | `nav_msgs/OccupancyGrid` | 2D probabilistic occupancy grid (when enabled) |
| `dlio/map_node/map` | `sensor_msgs/PointCloud2` | Accumulated map (prior + live keyframes) |
| `dlio/graph_slam/corrected_path` | `nav_msgs/Path` | Loop-closure corrected path |
| `dlio/graph_slam/corrected_map` | `sensor_msgs/PointCloud2` | Corrected full map |
| `dlio/graph_slam/corrected_kf_poses` | `geometry_msgs/PoseArray` | Corrected keyframe poses |
| `dlio/graph_slam/loop_closures` | `visualization_msgs/MarkerArray` | Loop closure visualization |

### TF Transforms

| Parent | Child | Description |
|--------|-------|-------------|
| `map` | `odom` | Global correction from continuous localization (when enabled) |
| `odom` | `base_link` | Odometry pose (smooth, continuous) |
| `base_link` | `imu` | Static IMU extrinsics |
| `base_link` | `lidar` | Static LiDAR extrinsics |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `pointcloud` | `sensor_msgs/PointCloud2` | Input LiDAR scan (remapped via launch arg) |
| `imu` | `sensor_msgs/Imu` | Input IMU data (remapped via launch arg) |

## Key Changes from Original DLIO

- **Modular code structure**: `OdomNode` split into 6 focused source files + shared utility modules (`scan_context.h`, `kfdb_io`)
- **Composable node architecture**: All nodes in a single process with IPC, replacing separate executables
- **NDT-OMP registration**: Optional NDT backend via `registration_method:=ndt` (default remains GICP)
- **C++ 17** (was C++ 14)
- **g2o dependency** for pose graph optimization
- **Custom `KeyframeStamped` message** for passing keyframe data between nodes
- **nanoflann `radiusSearch` bugfix**: Original used incorrect return value from `findNeighbors()`
- **TF/Path publishing fix**: Uses GICP-corrected `lidarPose` instead of raw `state` for consistent transforms
- **`length_traversed` fix**: Now updated in main loop instead of only in debug function (was preventing deskewed cloud publishing when `waitUntilMove: true`)

## Based On

This project is built on top of **Direct LiDAR-Inertial Odometry (DLIO)** by the [Verifiable & Control-Theoretic Robotics (VECTR) Lab](https://vectr.ucla.edu/) at UCLA.

```bibtex
@article{chen2022dlio,
  title={Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction},
  author={Chen, Kenny and Nemiroff, Ryan and Lopez, Brett T},
  journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={3983-3989},
  doi={10.1109/ICRA48891.2023.10160508}
}
```

Original repository: [https://github.com/vectr-ucla/direct_lidar_inertial_odometry](https://github.com/vectr-ucla/direct_lidar_inertial_odometry)

## Acknowledgements

- [DLIO](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) — Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez (UCLA VECTR Lab)
- [FastGICP](https://github.com/SMRT-AIST/fast_gicp) — Kenji Koide et al.
- [NanoFLANN](https://github.com/jlblancoc/nanoflann) — Jose Luis Blanco and Pranjal Kumar Rai
- [Scan Context](https://github.com/irapkaist/scancontext) — Giseop Kim and Ayoung Kim (KAIST)
- [g2o](https://github.com/RainerKuemmerle/g2o) — Rainer Kuemmerle et al.
- [NDT-OMP](https://github.com/koide3/ndt_omp) — Kenji Koide et al.

## License

This work is licensed under the terms of the MIT license.
