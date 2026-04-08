# DLIO-SCLC: Direct LiDAR-Inertial Odometry with Scan Context Loop Closure

An extended version of [Direct LiDAR-Inertial Odometry (DLIO)](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) with **LIO-SAM-style map optimization**, **Scan Context relocalization**, **prior map localization**, and **multiple registration backends**.

## What's New

| Feature | Description |
|---------|-------------|
| **LIO-SAM Map Optimization** | GTSAM iSAM2 incremental pose graph with loop closure (SC++ & distance-based) + optional GPS factors |
| **Multiple Registration Backends** | GICP, NDT, and Robust ICP (KISS-ICP-style point-to-point with Geman-McClure kernel) |
| **Voxel Hash Map** | KISS-ICP-style accumulated voxel map as alternative to KNN keyframe submap |
| **Pure LiDAR Mode** | `use_imu: false` — constant velocity model, no IMU dependency |
| **Scan Context Relocalization** | Automatic initial pose estimation using SC descriptors + GICP/NDT/RobustICP refinement |
| **Keyframe Database (KFDB)** | Persistent SC descriptors for relocalization (auto-saved, corrected KFDB from lio_sam_opt) |
| **Prior Map Localization** | Load map and localize with continuous + submap localization |
| **Continuous Localization** | Periodic global alignment with `map->odom` TF correction |
| **Submap Localization** | Two-stage submap-based localization with motion validation |
| **Registration Helper** | Shared utility for switchable registration across all localization paths |
| **Occupancy Grid Map** | Probabilistic 2D grid from LiDAR scans (Bresenham + log-odds Bayesian) |
| **Composable Nodes** | OdomNode + MapNode in one process, LIO-SAM opt in separate process |
| **RPY Extrinsics** | Specify IMU/LiDAR extrinsics as roll/pitch/yaw degrees instead of rotation matrices |
| **Robosense Support** | Auto-detect Robosense (BPearl, etc.) timestamp format for deskewing |

## Architecture

```
┌───────────────────────────────────────────┐
│         dlio_container (process 1)        │
│                                           │
│  ┌──────────────┐   ┌──────────────┐     │
│  │   OdomNode   │──>│   MapNode    │     │
│  │              │   │              │     │
│  │ - IMU fusion │   │ - Map accum  │     │
│  │ - GICP/NDT/  │   │ - Auto-save  │     │
│  │   RobustICP  │   │ - PCD I/O    │     │
│  │ - Relocalize │   │              │     │
│  │ - KFDB save  │   └──────────────┘     │
│  └──────────────┘                         │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│    lio_sam_opt_container (process 2)      │
│                                           │
│  ┌─────────────────────────────────┐     │
│  │  LioSamMapOptimizationNode     │     │
│  │                                 │     │
│  │ - GTSAM iSAM2 pose graph       │     │
│  │ - Loop closure (SC++ + dist)    │     │
│  │ - GPS factor integration        │     │
│  │ - map->odom TF publishing       │     │
│  │ - Corrected map/path publish    │     │
│  └─────────────────────────────────┘     │
└───────────────────────────────────────────┘
```

LIO-SAM opt runs in a **separate process** to prevent heavy ICP/GTSAM work from starving the real-time odometry pipeline.

## Code Structure

```
include/dlio/
├── odom.h                       # OdomNode class declaration
├── map.h                        # MapNode class declaration
├── lio_sam_map_optimization.h   # LioSamMapOptimizationNode class
├── robust_icp.h                 # Point-to-point ICP with GM kernel (KISS-ICP style)
├── voxel_hash_map.h             # Voxel hash map (KISS-ICP style local map)
├── registration_helper.h        # Unified registration wrapper (GICP/NDT/RobustICP)
├── scan_context.h               # Scan Context utilities (header-only)
├── kfdb_io.h                    # Keyframe Database I/O interface
├── occupancy_grid.h             # Probabilistic occupancy grid generator
├── utils.h                      # Common types and utilities
└── dlio.h                       # Package-wide defines

src/dlio/
├── odom.cc                      # Constructor, getParams(), start()
├── odom_callbacks.cc            # callbackPointCloud, callbackImu, deskewing
├── odom_registration.cc         # GICP/NDT/RobustICP alignment, IMU integration
├── odom_keyframes.cc            # Keyframe management, submap/voxel map building
├── odom_relocalization.cc       # Prior map, SC database, relocalization, KFDB
├── odom_services.cc             # Services, publishing, continuous localization
├── odom_submap_localization.cc  # Two-stage submap localization
├── robust_icp.cc                # RobustICP implementation
├── voxel_hash_map.cc            # VoxelHashMap implementation
├── registration_helper.cc       # RegistrationHelper implementation
├── occupancy_grid.cc            # Occupancy grid generator
├── kfdb_io.cc                   # KFDB binary file read/write
├── lio_sam_map_optimization.cc  # LIO-SAM map optimization implementation
├── map.cc                       # MapNode implementation
└── map_node.cc                  # MapNode component registration
```

## Dependencies

- Ubuntu 22.04
- ROS 2 Humble
- C++ 17
- Point Cloud Library >= 1.10.0
- Eigen >= 3.3.7
- GTSAM >= 4.1
- GeographicLib (for GPS support)
- [ndt_omp](https://github.com/koide3/ndt_omp)
- OpenMP >= 4.5

```bash
sudo apt install libomp-dev libpcl-dev libeigen3-dev ros-humble-pcl-ros
# GTSAM: build from source or install via PPA
```

## Build

```bash
cd <your_ws>
colcon build --packages-select direct_lidar_inertial_odometry
source install/setup.bash
```

## Usage

### Mapping

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  map_mode:=mapping \
  map_path:=/path/to/my_map.pcd \
  pointcloud_topic:=/your/pointcloud \
  imu_topic:=/your/imu
```

Produces:
- `my_map.pcd` — point cloud map
- `my_map.kfdb` — keyframe database for relocalization
- `my_map_corrected.pcd` — loop-closure corrected map (from lio_sam_opt)
- `my_map_corrected.kfdb` — corrected keyframe database

### Localization

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  map_mode:=localization \
  map_path:=/path/to/my_map.pcd \
  relocalize:=true \
  pointcloud_topic:=/your/pointcloud \
  imu_topic:=/your/imu
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `map_mode` | `localization` | `mapping` or `localization` |
| `map_path` | `""` | Path to PCD map file |
| `relocalize` | `true` | Enable SC relocalization |
| `use_corrected` | `true` | Load `_corrected` map/KFDB files |
| `pointcloud_topic` | `points_raw` | Input point cloud topic |
| `imu_topic` | `imu_raw` | Input IMU topic |
| `gps_topic` | `gps_raw` | GPS NavSatFix topic |
| `rviz` | `false` | Launch RViz |

## Configuration

### Registration Methods (`params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/registration_method` | `gicp` | Odometry: `gicp`, `ndt`, or `robust_icp` |
| `odom/localization/registration_method` | `gicp` | Localization: `gicp`, `ndt`, or `robust_icp` |

### Submap Strategy (`params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/submap/method` | `keyframe` | `keyframe` (KNN) or `voxel_hash_map` (KISS-ICP style) |
| `odom/submap/keyframe/knn` | `10` | Nearest keyframes for submap |
| `odom/submap/voxel_hash_map/voxel_size` | `1.0` | Voxel size (m) |
| `odom/submap/voxel_hash_map/max_distance` | `100.0` | Max distance from origin (m) |
| `odom/submap/voxel_hash_map/max_points_per_voxel` | `20` | Points per voxel |

### IMU Settings (`params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/use_imu` | `true` | Enable IMU (`false` = pure LiDAR, constant velocity model) |
| `odom/use_2d_imu` | `false` | Constrain IMU to 2D (zero roll/pitch gyro, replace accel Z with gravity) |

### Extrinsics (`dlio.yaml`)

```yaml
# RPY in degrees (preferred) — set to [0,0,0] to use R matrix instead
extrinsics/baselink2imu/rpy: [0.0, 0.0, 0.0]
extrinsics/baselink2lidar/rpy: [0.0, 0.0, 0.0]
```

### LIO-SAM Map Optimization (`lio_sam_map_optimization.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loop_closure/search_radius` | `15.0` | Spatial search radius (m) |
| `loop_closure/fitness_score_threshold` | `0.3` | GICP fitness threshold |
| `registration_method` | `icp` | LC registration: `icp`, `gicp`, `ndt`, `robust_icp` |
| `odom_noise/rotation` | `0.05` | Odometry rotation variance |
| `odom_noise/translation` | `0.3` | Odometry translation variance |
| `loop_noise/multiplier` | `0.005` | LC noise = fitness * this |
| `map/tf_source` | (from params.yaml) | Who publishes `map->odom` TF |

### Sensor Support

Auto-detected from point cloud fields:

| Sensor | Field | Detection |
|--------|-------|-----------|
| Ouster | `t` | Nanosecond offset |
| Velodyne | `time` | Relative seconds |
| Hesai | `timestamp` (> 1e6) | Absolute seconds |
| Robosense | `timestamp` (< 1e6) | Relative seconds |
| Livox | `timestamp` (> 1e14) | Nanoseconds epoch |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `dlio/odom_node/odom` | `nav_msgs/Odometry` | Odometry estimate |
| `dlio/odom_node/pose` | `geometry_msgs/PoseStamped` | Current pose |
| `dlio/odom_node/path` | `nav_msgs/Path` | Trajectory path |
| `dlio/odom_node/pointcloud/deskewed` | `sensor_msgs/PointCloud2` | Deskewed scan |
| `dlio/odom_node/pointcloud/keyframe` | `sensor_msgs/PointCloud2` | Keyframe cloud |
| `dlio/odom_node/keyframes` | `geometry_msgs/PoseArray` | Keyframe poses |
| `dlio/odom_node/occupancy_grid` | `nav_msgs/OccupancyGrid` | 2D occupancy grid |
| `dlio/map_node/map` | `sensor_msgs/PointCloud2` | Accumulated map |
| `dlio/lio_sam_opt/corrected_path` | `nav_msgs/Path` | Loop-closure corrected path |
| `dlio/lio_sam_opt/corrected_map` | `sensor_msgs/PointCloud2` | Corrected full map |
| `dlio/lio_sam_opt/corrected_kf_poses` | `geometry_msgs/PoseArray` | Corrected keyframe poses |
| `dlio/lio_sam_opt/loop_closures` | `visualization_msgs/MarkerArray` | Loop closure markers |

### TF Transforms

| Parent | Child | Description |
|--------|-------|-------------|
| `map` | `odom` | Global correction (from lio_sam_opt or continuous localization) |
| `odom` | `base_link` | Odometry pose (smooth, continuous) |
| `base_link` | `imu` | Static IMU extrinsics |
| `base_link` | `lidar` | Static LiDAR extrinsics |

## Runtime Services

| Service | Topic | Description |
|---------|-------|-------------|
| `GetState` | `/dlio/odom_node/get_state` | Query state, pose, stats |
| `SetMode` | `/dlio/odom_node/set_mode` | Switch mapping/localization |
| `SetPose` | `/dlio/odom_node/set_pose` | Manually set robot pose |
| `Relocalize` | `/dlio/odom_node/relocalize` | Trigger SC+GICP relocalization |
| `NewMap` | `/dlio/odom_node/new_map` | Clear everything, start fresh |
| `NewMapWZero` | `/dlio/odom_node/new_map_w_zero` | Clear + reset pose to zero |
| `SavePCD` | `/dlio/map_node/save_pcd` | Save current map |
| `SaveCorrectedPCD` | `/dlio/lio_sam_opt/save_corrected_pcd` | Save loop-closure corrected map |

## Based On

This project is built on **Direct LiDAR-Inertial Odometry (DLIO)** by the [VECTR Lab](https://vectr.ucla.edu/) at UCLA.

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

## Acknowledgements

- [DLIO](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) — Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez (UCLA VECTR Lab)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp) — Ignacio Vizzo et al. (inspiration for Robust ICP + Voxel Hash Map)
- [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) — Tixiao Shan (inspiration for iSAM2 map optimization)
- [FastGICP](https://github.com/SMRT-AIST/fast_gicp) — Kenji Koide et al.
- [NanoFLANN](https://github.com/jlblancoc/nanoflann) — Jose Luis Blanco
- [Scan Context](https://github.com/irapkaist/scancontext) — Giseop Kim and Ayoung Kim (KAIST)
- [GTSAM](https://gtsam.org/) — Frank Dellaert et al. (Georgia Tech)
- [NDT-OMP](https://github.com/koide3/ndt_omp) — Kenji Koide et al.

## License

This work is licensed under the terms of the MIT license.
