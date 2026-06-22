# C_LIO: Compact LiDAR-Inertial Odometry

**C_LIO** (Compact LiDAR-Inertial Odometry) is an evolution of [DLIO (Direct LiDAR-Inertial Odometry)](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) by the UCLA VECTR Lab. While DLIO provides a lightweight, real-time LIO frontend with continuous-time motion correction, C_LIO builds on top of it into a complete, production-ready SLAM system:

- **Backend pose graph** — LIO-SAM-style iSAM2 with loop closure (distance + Scan Context++) and optional GPS factors running in a separate process, so pose-graph correction never starves the real-time odometry pipeline
- **Appearance-based auto localization** — Scan Context++ relocalization for automatic initial pose estimation from a prior map, plus continuous Bayesian global alignment and two-stage submap localization for GPS-denied environments
- **Multiple registration backends** — six interchangeable backends (`gicp`, `ndt`, `robust_icp`, and their CUDA counterparts) unified under a single `align()` API, selectable at runtime for both odometry and loop closure
- **Pluggable fusion engine** — switchable IMU/scan-match fusion strategies (geometric observer, Kalman filter, error-state EKF) with a pose safety gate that rejects physically implausible results
- **CUDA acceleration** — GPU kernels for deskewing, voxel filtering, GICP/RobustICP/NDT registration, and nearest-neighbor search; automatic CPU fallback when CUDA is absent
- **Keyframe database (KFDB)** — persistent appearance descriptors and body-frame scans saved to disk for session-to-session relocalization
- **Occupancy grid, multi-sensor support, composable ROS 2 nodes**, and more

## Features

| Feature | Description |
|---------|-------------|
| **CUDA Acceleration** | GPU backends for registration (GICP/NDT/RobustICP), preprocessing (deskew + voxel filter), and nearest-neighbor search. Auto-detected at build time; CPU fallback when CUDA is absent |
| **Registration Engine (6 backends)** | `gicp`, `gicp_cuda`, `ndt`, `ndt_cuda`, `robust_icp`, `robust_icp_cuda` — unified one-shot `align()` API |
| **Fusion Engine** | Switchable IMU/scan-match fusion: Geometric Observer (`geo`), Kalman Filter (`kf`), Error-State EKF (`ekf`) |
| **Pose Safety Gate** | Rejects physically implausible scan-match results (jump/fitness/spaciousness), falls back to IMU-only pose |
| **Motion Model Constraints** | Constrain IMU propagation to `ackermann`, `diff_drive`, or `holonomic` motion |
| **Prefilter Engine** | Optional noise removal after deskew: `sor`, `sor_cuda`, `ror`, `dror` (distance-adaptive) |
| **Appearance Engine** | Pluggable place descriptors for loop closure / relocalization: `sc++` (active), `std` (planned), CUDA variants |
| **External Odometry** | Fuse wheel encoder / INS twist (`TwistWithCovarianceStamped`) as velocity source instead of IMU accel |
| **LIO-SAM Map Optimization** | GTSAM iSAM2 incremental pose graph with loop closure (distance + SC++) + optional GPS factors + periodic batch LM |
| **GTSAM IMU Preintegration** | `simple` dead reckoning or GTSAM `PreintegratedImuMeasurements` |
| **Voxel Hash Map** | KISS-ICP-style accumulated voxel map as alternative to KNN keyframe submap |
| **Pure LiDAR Mode** | `use_imu: false` — constant velocity model, no IMU dependency |
| **Scan Context Relocalization** | Automatic initial pose estimation using SC descriptors + GICP/NDT/RobustICP refinement |
| **Keyframe Database (KFDB v5)** | Persistent appearance descriptors + body-frame scans for relocalization (auto-saved, corrected KFDB from lio_sam_opt) |
| **Prior Map Localization** | Load map and localize with continuous + submap localization |
| **Continuous Localization** | Bayesian periodic global alignment with `map->odom` TF correction + g2o verification |
| **Submap Localization** | GPS-denied two-stage submap localization with motion validation |
| **GPS Fusion** | NavSatFix → ENU factors with auto yaw calibration, GPS-based loop closure |
| **Occupancy Grid Map** | Probabilistic 2D grid from LiDAR scans (Bresenham + log-odds Bayesian) |
| **Composable Nodes** | OdomNode (with map accumulation) in one process, LIO-SAM opt in a separate process |
| **RPY Extrinsics** | Specify IMU/LiDAR/GPS extrinsics as roll/pitch/yaw degrees instead of rotation matrices |
| **Multi-Sensor Support** | Auto-detect Ouster / Velodyne / Hesai / Robosense / Livox timestamp formats for deskewing |

## Architecture

```
┌───────────────────────────────────────────────┐
│           c_lio_container (process 1)          │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │            OdomNode                  │    │
│  │                                     │    │
│  │  - GPU/CPU preprocess (deskew+vox)  │    │
│  │  - PrefilterEngine (noise removal)  │    │
│  │  - RegistrationEngine (GICP/NDT/RI) │    │
│  │  - FusionEngine (geo/kf/ekf) + gate │    │
│  │  - Keyframe + submap / voxel map    │    │
│  │  - Map accumulation + auto-save     │    │
│  │  - Relocalization + KFDB I/O        │    │
│  │  - Continuous / submap localization │    │
│  │  - Occupancy grid                   │    │
│  └─────────────────────────────────────┘    │
└───────────────────────────────────────────────┘
                    │ keyframe_stamped / odom
                    ▼
┌───────────────────────────────────────────────┐
│      lio_sam_opt_container (process 2)        │
│              (prefix: nice -n 10)             │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │     LioSamMapOptimizationNode        │    │
│  │                                     │    │
│  │  - GTSAM iSAM2 pose graph + batch LM │    │
│  │  - Loop closure (distance + SC++)   │    │
│  │  - GPS factor integration            │    │
│  │  - map->odom TF publishing           │    │
│  │  - Corrected map/path/KF publish     │    │
│  │  - Fused corrected pose (KF)         │    │
│  └─────────────────────────────────────┘    │
└───────────────────────────────────────────────┘
```

LIO-SAM opt runs in a **separate process** (`nice -n 10`) to prevent heavy ICP/GTSAM work from starving the real-time odometry pipeline. Map accumulation now lives inside OdomNode (the old standalone MapNode was merged in).

## Code Structure

```
include/c_lio/
├── c_lio.h                          # Package-wide defines (C_LIO_HAS_CUDA, PointType, etc.)
├── odom/
│   ├── odom.h                      # OdomNode class declaration
│   ├── scan_context.h              # Scan Context utilities (header-only impl)
│   ├── kfdb_io.h                   # Keyframe Database I/O interface
│   ├── occupancy_grid.h            # Probabilistic occupancy grid generator
│   └── utils.h                     # Common types and utilities
├── engines/
│   ├── registration_engine.h       # Unified registration (GICP/NDT/RobustICP + CUDA)
│   ├── appearance_engine.h         # Place descriptors (SC++ / STD)
│   └── prefilter_engine.h          # Point cloud noise prefilter (SOR/ROR/DROR)
├── algorithms/
│   ├── nano_gicp/                  # NanoGICP + nanoflann KD-tree
│   ├── robust_icp.h                # Point-to-point ICP with GM kernel (KISS-ICP style)
│   ├── voxel_hash_map.h            # Voxel hash map (KISS-ICP style local map)
│   └── error_state_ekf.h           # Error-state EKF fusion
├── mapping/
│   └── lio_sam_map_optimization.h  # LioSamMapOptimizationNode class
└── cuda/
    ├── common.cuh, transforms.cuh  # CUDA helpers
    ├── nn_search.cuh               # GPU voxel-hash nearest neighbor
    ├── gicp_gpu_wrapper.h          # GICP GPU correspondence
    ├── linearize.cuh               # GICP GPU linearization
    ├── robust_icp_cuda.h           # Robust ICP CUDA
    ├── preprocess_cuda.cuh         # GPU deskew + voxel filter
    └── point_cloud_cuda.cuh        # GPU point cloud container

src/c_lio/
├── odom/
│   ├── odom.cc                     # Constructor, getParams(), start()
│   ├── odom_callbacks.cc           # callbackPointCloud, callbackImu, deskewing
│   ├── odom_registration.cc        # Registration + IMU integration + fusion dispatch
│   ├── odom_keyframes.cc           # Keyframe management, submap/voxel map building
│   ├── odom_relocalization.cc      # Prior map, SC database, relocalization, KFDB
│   ├── odom_services.cc            # Services, publishing, continuous localization
│   ├── odom_submap_localization.cc # Two-stage submap localization
│   ├── odom_node.cc                # OdomNode component registration + standalone main
│   ├── kfdb_io.cc                  # KFDB binary file read/write
│   └── occupancy_grid.cc           # Occupancy grid generator
├── engines/
│   ├── registration_engine.cc      # RegistrationEngine implementation
│   ├── appearance_engine.cc        # AppearanceEngine implementation
│   └── prefilter_engine.cc         # PrefilterEngine implementation
├── algorithms/
│   ├── nano_gicp/                  # NanoGICP / nanoflann sources
│   ├── robust_icp.cc               # RobustICP implementation
│   ├── voxel_hash_map.cc           # VoxelHashMap implementation
│   └── error_state_ekf.cc          # Error-state EKF implementation
├── mapping/
│   ├── lio_sam_map_optimization.cc       # LIO-SAM map optimization implementation
│   └── lio_sam_map_optimization_node.cc  # Component registration + standalone main
├── cuda/
│   ├── transforms.cu, nn_search.cu, linearize.cu
│   ├── gicp_gpu_wrapper.cu, robust_icp_cuda.cu
│   └── preprocess_cuda.cu          # GPU deskew + voxel filter
└── tools/
    ├── imu_integrator_node.cc      # Debug: pure IMU trajectory integrator
    └── imu_lidar_calib_node.cc     # IMU↔LiDAR extrinsic calibration (pure GICP)
```

## Dependencies

- Ubuntu 22.04
- ROS 2 Humble
- C++ 17
- Point Cloud Library >= 1.10.0
- Eigen >= 3.3.7
- GTSAM >= 4.1
- g2o (`ros-humble-libg2o`) — loop closure verification
- GeographicLib — GPS support
- [ndt_omp](https://github.com/koide3/ndt_omp) (`ndt_omp_ros2`)
- OpenMP >= 4.5
- **CUDA Toolkit** (optional) — enables GPU registration/preprocessing; CPU fallback if absent
- **`ndt_cuda_ros2`** (optional) — enables the `ndt_cuda` registration method

```bash
sudo apt install libomp-dev libpcl-dev libeigen3-dev ros-humble-pcl-ros ros-humble-libg2o
# GTSAM: build from source or install via PPA
# CUDA Toolkit: install via NVIDIA repo if GPU acceleration is desired
```

## Build

```bash
cd <your_ws>
# Build ndt_cuda_ros2 first if the ndt_cuda method is needed:
colcon build --packages-select ndt_cuda_ros2 c_lio
source install/setup.bash
```

CUDA is enabled automatically when `CUDAToolkit` is found (`C_LIO_HAS_CUDA`). The `ndt_cuda` method is enabled only when `ndt_cuda_ros2` is also found (`C_LIO_HAS_NDT_CUDA`); the two are decoupled so a broken `ndt_cuda_ros2` build does not disable C_LIO's own CUDA kernels. Build messages report which mode was selected.

## Usage

### Mapping

```bash
ros2 launch c_lio c_lio.launch.py \
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

> If `map_path` is empty it defaults to `$HOME/.ros/c_lio_map.pcd`.

### Localization

```bash
ros2 launch c_lio c_lio.launch.py \
  map_mode:=localization \
  map_path:=/path/to/my_map.pcd \
  relocalize:=true \
  pointcloud_topic:=/your/pointcloud \
  imu_topic:=/your/imu
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `map_mode` | `mapping` | `mapping` or `localization` |
| `map_path` | `""` | Path to PCD map file (load/save) |
| `relocalize` | `false` | Enable SC relocalization for initial pose |
| `use_corrected` | `true` | Load `_corrected` map/KFDB files in localization mode |
| `pointcloud_topic` | `points_raw` | Input point cloud topic |
| `imu_topic` | `imu_raw` | Input IMU topic |
| `gps_topic` | `gps_raw` | GPS NavSatFix topic |
| `ext_odom_topic` | `/odom` | External odometry / twist topic |
| `rviz` | `false` | Launch RViz |

## Configuration

Parameters are split across seven YAML files in `cfg/` (loaded by the launch file):

| File | Scope |
|------|-------|
| `c_lio.yaml` | General: version, frames, debug, `map/tf_source` |
| `sensor.yaml` | IMU, LiDAR, extrinsics, external odom, calibration |
| `odom.yaml` | Preprocessing, prefilter, registration, keyframes, submap |
| `fusion.yaml` | Fusion strategy, KF/EKF noise, safety gate, motion model |
| `map.yaml` | Map mode, relocalization, continuous/submap localization, GPS |
| `occupancy_grid.yaml` | Occupancy grid generation |
| `lio_sam_map_optimization.yaml` | iSAM2 pose graph, loop closure, GPS factors |

### Preprocessing & Registration (`odom.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/preprocessing/gpu` | `true` | GPU deskew + voxel grid filter (requires CUDA) |
| `odom/preprocessing/voxelFilter/res` | `1.0` | Voxel leaf size (m) |
| `odom/preprocessing/cropBoxFilter/size` | `3.0` | Ego crop box size (m) |
| `odom/prefilter/method` | `none` | Noise prefilter: `none`, `sor`, `sor_cuda`, `ror`, `dror` |
| `odom/registration_method` | `gicp_cuda` | Odometry registration backend (see below) |
| `odom/localization/registration_method` | `gicp` | Registration used in localization mode |

**Registration backends** (`odom/registration_method`): `gicp`, `gicp_cuda`, `ndt`, `ndt_cuda`, `robust_icp`, `robust_icp_cuda`. CUDA variants require a CUDA build; `ndt_cuda` additionally requires `ndt_cuda_ros2`.

### Submap Strategy (`odom.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/submap/method` | `keyframe` | `keyframe` (KNN + hull), `voxel_hash_map`, or `recent_keyframes` |
| `odom/submap/keyframe/knn` | `15` | Nearest keyframes for submap |
| `odom/submap/recent/n` | `30` | Number of recent keyframes (`recent_keyframes` mode) |
| `odom/submap/voxel_hash_map/voxel_size` | `0.75` | Voxel size (m) |
| `odom/submap/voxel_hash_map/max_distance` | `200.0` | Max distance from origin (m) |
| `odom/submap/voxel_hash_map/max_points_per_voxel` | `40` | Points per voxel |

### Fusion & Safety (`fusion.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/fusion/method` | `kf` | `geo` (geometric observer), `kf` (Kalman), `ekf` (error-state EKF) |
| `odom/fusion/kf/sigma_*` | — | KF process/measurement noise (IMU vs scan-match trust) |
| `odom/fusion/ekf/*` | — | EKF noise, Mahalanobis gate, fitness/degeneracy weights |
| `odom/fusion/gate/enabled` | `false` | Reject physically implausible scan-match results |
| `odom/fusion/gate/max_translation` | `5.0` | Max position jump per scan (m) |
| `odom/fusion/gate/max_rotation_deg` | `45.0` | Max rotation jump per scan (deg) |
| `motion_model/type` | `none` | `none`, `ackermann`, `diff_drive`, `holonomic` |

### IMU & Extrinsics (`sensor.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odom/use_imu` | `true` | Enable IMU (`false` = pure LiDAR, constant velocity model) |
| `odom/use_2d_imu` | `false` | Constrain IMU to 2D (zero roll/pitch gyro, accel Z = gravity) |
| `odom/imu/preintegration` | `gtsam` | `simple` dead reckoning or `gtsam` PreintegratedImuMeasurements |
| `odom/imu/gravityRemoved` | `false` | IMU driver already removed gravity from accel |
| `odom/imu/accelInG` | `false` | IMU publishes accel in g instead of m/s² |
| `odom/external_odom/enabled` | `true` | Fuse external twist (wheel/INS) as velocity source |
| `odom/external_odom/topic` | `/applanix/lvx_client/twist` | `TwistWithCovarianceStamped` topic |

Extrinsics use RPY in degrees (set RPY to `[0,0,0]` to fall back to identity/R-matrix):

```yaml
extrinsics/baselink2imu/t:   [0.0, 0.0, 0.0]
extrinsics/baselink2imu/rpy: [0.0, 0.0, 0.0]
extrinsics/baselink2lidar/t:   [0.0, 0.0, 0.0]
extrinsics/baselink2lidar/rpy: [0.0, 0.0, 0.0]
extrinsics/baselink2gps/t:   [0.0, 0.0, 0.0]
extrinsics/baselink2gps/rpy: [0.0, 0.0, 0.0]   # ENU→odom yaw auto-calibrated at startup
```

### Relocalization & Localization (`map.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map/appearance_method` | `sc++` | Place descriptor: `sc++`, `sc++_cuda`, `std`, `std_cuda` |
| `map/continuous_localize` | `true` | Bayesian periodic global alignment with `map->odom` correction |
| `map/continuous_localize/g2o_verification` | `true` | g2o chi² verification of continuous loc matches |
| `map/submap_localize` | `true` | GPS-denied two-stage submap localization |
| `gps/enabled` | `true` | Enable GPS NavSatFix fusion |

### LIO-SAM Map Optimization (`lio_sam_map_optimization.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map_correction` | `false` | Master switch for loop closure + GPS correction |
| `registration_method` | `gicp_cuda` | LC registration: `gicp`/`gicp_cuda`/`ndt`/`ndt_cuda`/`robust_icp`/`robust_icp_cuda` |
| `loop_closure/search_radius` | `15.0` | Spatial search radius (m) |
| `loop_closure/fitness_score_threshold` | `0.3` | GICP fitness threshold |
| `sc/enabled` | `true` | Scan Context++ as secondary/fallback loop closure |
| `gps/enabled` | `true` | Add GPS factors to the pose graph |
| `isam/batch_optimization` | `true` | Run periodic LM batch optimization |
| `odom_noise/rotation` / `translation` | `0.2` / `9.0` | Odometry factor variances |
| `loop_noise/multiplier` | `0.01` | LC noise = fitness × this |

> `map/tf_source` (in `c_lio.yaml`) decides who publishes `map->odom`: `odom` (OdomNode) or `lio_sam_opt`. Do **not** override it in `lio_sam_map_optimization.yaml`, or both nodes will publish TF and cause jumping.

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
| `c_lio/odom_node/odom` | `nav_msgs/Odometry` | Odometry estimate |
| `c_lio/odom_node/pose` | `geometry_msgs/PoseStamped` | Current pose |
| `c_lio/odom_node/path` | `nav_msgs/Path` | Trajectory path |
| `c_lio/odom_node/pointcloud/deskewed` | `sensor_msgs/PointCloud2` | Deskewed scan |
| `c_lio/odom_node/pointcloud/deskewed_raw` | `sensor_msgs/PointCloud2` | Deskewed scan (pre-voxel) |
| `c_lio/odom_node/pointcloud/keyframe` | `sensor_msgs/PointCloud2` | Keyframe cloud |
| `c_lio/odom_node/keyframes` | `geometry_msgs/PoseArray` | Keyframe poses |
| `c_lio/odom_node/keyframe_stamped` | `KeyframeStamped` | Keyframe → lio_sam_opt |
| `c_lio/odom_node/occupancy_grid` | `nav_msgs/OccupancyGrid` | 2D occupancy grid |
| `localization_confidence` | `std_msgs/Float32` | Localization confidence |
| `c_lio/map_node/map` | `sensor_msgs/PointCloud2` | Accumulated map |
| `c_lio/lio_sam_opt/corrected_path` | `nav_msgs/Path` | Loop-closure corrected path |
| `c_lio/lio_sam_opt/corrected_map` | `sensor_msgs/PointCloud2` | Corrected full map |
| `c_lio/lio_sam_opt/corrected_kf_poses` | `geometry_msgs/PoseArray` | Corrected keyframe poses |
| `c_lio/lio_sam_opt/corrected_fusion_path` | `nav_msgs/Path` | KF-fused corrected path |
| `c_lio/lio_sam_opt/corrected_fusion_odom` | `nav_msgs/Odometry` | KF-fused corrected odom |
| `c_lio/lio_sam_opt/loop_closures` | `visualization_msgs/MarkerArray` | Loop closure markers |

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
| `GetState` | `/c_lio/odom_node/get_state` | Query state, pose, stats |
| `SetMode` | `/c_lio/odom_node/set_mode` | Switch mapping/localization |
| `SetPose` | `/c_lio/odom_node/set_pose` | Manually set robot pose |
| `Relocalize` | `/c_lio/odom_node/relocalize` | Trigger SC + registration relocalization |
| `NewMap` | `/c_lio/odom_node/new_map` | Clear everything, start fresh |
| `NewMapWZero` | `/c_lio/odom_node/new_map_w_zero` | Clear + reset pose to zero |
| `SavePCD` | `/c_lio/map_node/save_pcd` | Save current map |
| `SavePCD` | `/c_lio/lio_sam_opt/save_corrected_pcd` | Save loop-closure corrected map |

## Debug Tools

| Executable | Description |
|------------|-------------|
| `imu_integrator_node` | Integrates raw IMU into a pure-IMU trajectory for sanity checks |
| `imu_lidar_calib_node` | IMU↔LiDAR extrinsic calibration via pure GICP (standalone, no SLAM dependency) |

Standalone (non-composable) executables `c_lio_odom_node` and `c_lio_lio_sam_map_opt_node` are also built as convenience wrappers.

## Based On

C_LIO is built on top of **DLIO (Direct LiDAR-Inertial Odometry)** by the [VECTR Lab](https://vectr.ucla.edu/) at UCLA.

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
- [g2o](https://github.com/RainerKuemmerle/g2o) — Rainer Kümmerle et al.
- [NDT-OMP](https://github.com/koide3/ndt_omp) — Kenji Koide et al.

## License

This work is licensed under the terms of the MIT license.
</content>
