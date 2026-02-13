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

## Dependencies

- Ubuntu 22.04
- ROS 2 Humble
- C++ 17
- Point Cloud Library >= 1.10.0
- Eigen >= 3.3.7
- g2o (via `ros-humble-libg2o`)
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
| `map_mode` | `mapping` | `mapping` or `localization` |
| `map_path` | `""` | Path to PCD map file (save/load) |
| `relocalize` | `false` | Enable Scan Context relocalization |
| `pointcloud_topic` | `points_raw` | Input point cloud topic |
| `imu_topic` | `imu_raw` | Input IMU topic |
| `rviz` | `false` | Launch RViz |

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

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `dlio/odom_node/odom` | `nav_msgs/Odometry` | Odometry estimate |
| `dlio/odom_node/pose` | `geometry_msgs/PoseStamped` | Current pose |
| `dlio/odom_node/path` | `nav_msgs/Path` | Trajectory path |
| `dlio/odom_node/pointcloud/deskewed` | `sensor_msgs/PointCloud2` | Deskewed scan in world frame |
| `dlio/odom_node/pointcloud/keyframe` | `sensor_msgs/PointCloud2` | Keyframe cloud |
| `dlio/odom_node/keyframes` | `geometry_msgs/PoseArray` | All keyframe poses |
| `dlio/graph_slam/corrected_path` | `nav_msgs/Path` | Loop-closure corrected path |
| `dlio/graph_slam/corrected_map` | `sensor_msgs/PointCloud2` | Corrected full map |
| `dlio/graph_slam/corrected_kf_poses` | `geometry_msgs/PoseArray` | Corrected keyframe poses |
| `dlio/graph_slam/loop_closures` | `visualization_msgs/MarkerArray` | Loop closure visualization |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `pointcloud` | `sensor_msgs/PointCloud2` | Input LiDAR scan (remapped via launch arg) |
| `imu` | `sensor_msgs/Imu` | Input IMU data (remapped via launch arg) |

## Key Changes from Original DLIO

- **Composable node architecture**: All nodes in a single process with IPC, replacing separate executables
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

## License

This work is licensed under the terms of the MIT license.
