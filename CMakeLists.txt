###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.18)
project(direct_lidar_inertial_odometry)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# ---- Dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(ndt_omp_ros2 REQUIRED)
find_package(ndt_cuda_ros2 QUIET)
find_package(GeographicLib REQUIRED)

# g2o (from ros-humble-libg2o)
set(g2o_DIR "/opt/ros/humble/lib/cmake/g2o")
find_package(g2o REQUIRED COMPONENTS g2o::core g2o::types_slam3d g2o::solver_eigen)
add_compile_definitions(G2O_USE_VENDORED_CERES)

# GTSAM (for LIO-SAM-style map optimization)
find_package(GTSAM REQUIRED QUIET)

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message("ERROR: OpenMP could not be found.")
endif()

# ---- CUDA (optional) ----
# DLIO_HAS_CUDA      → DLIO-native CUDA kernels (transforms.cu, gicp_gpu_wrapper.cu,
#                      preprocess_cuda.cu, robust_icp_cuda.cu, nn_search.cu, linearize.cu).
#                      Only requires CUDAToolkit.
# DLIO_HAS_NDT_CUDA  → Optional NDT_CUDA registration method via external ndt_cuda_ros2
#                      package. Decoupled so a broken ndt_cuda_ros2 build does not
#                      disable DLIO's own CUDA acceleration.
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  set(DLIO_HAS_CUDA ON)
  link_directories(${CUDAToolkit_LIBRARY_DIR})
  if(ndt_cuda_ros2_FOUND)
    set(DLIO_HAS_NDT_CUDA ON)
    add_compile_definitions(DLIO_HAS_NDT_CUDA=1)
    message(STATUS "DLIO CUDA: ON (native + NDT_CUDA method)")
  else()
    set(DLIO_HAS_NDT_CUDA OFF)
    message(STATUS "DLIO CUDA: ON (native only — ndt_cuda_ros2 missing; NDT_CUDA method unavailable)")
  endif()
else()
  set(DLIO_HAS_CUDA OFF)
  set(DLIO_HAS_NDT_CUDA OFF)
  message(STATUS "CUDAToolkit not found — building CPU-only")
endif()

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

# ---- Includes ----
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GeographicLib_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  /opt/ros/humble/include
)
link_directories(
  ${GTSAM_LIBRARY_DIRS}
)
link_directories(
  ${PCL_LIBRARY_DIRS}
  /opt/ros/humble/lib/x86_64-linux-gnu
)

# ---- Message/Service Generation ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyframeStamped.msg"
  "srv/SavePCD.srv"
  "srv/SetMode.srv"
  "srv/Relocalize.srv"
  "srv/SetPose.srv"
  "srv/GetState.srv"
  "srv/NewMap.srv"
  "srv/NewMapWZero.srv"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

# ---- Internal Libraries ----

# NanoFLANN
add_library(nanoflann STATIC
  src/dlio/algorithms/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/dlio/algorithms/nano_gicp/lsq_registration.cc
  src/dlio/algorithms/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)

# ---- CUDA (conditional, after rosidl to avoid linker language conflicts) ----
if(DLIO_HAS_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

  add_library(dlio_cuda STATIC
    src/dlio/cuda/transforms.cu
    src/dlio/cuda/nn_search.cu
    src/dlio/cuda/linearize.cu
    src/dlio/cuda/gicp_gpu_wrapper.cu
    src/dlio/cuda/robust_icp_cuda.cu
    src/dlio/cuda/preprocess_cuda.cu
  )
  target_include_directories(dlio_cuda PUBLIC include ${PCL_INCLUDE_DIRS})
  target_link_libraries(dlio_cuda CUDA::cudart)
  target_compile_options(dlio_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20013>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20015>
  )
  set_target_properties(dlio_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

# ---- Compile definitions for DLIO_HAS_CUDA ----
if(DLIO_HAS_CUDA)
  set(_DLIO_CUDA_DEF DLIO_HAS_CUDA=1)
else()
  set(_DLIO_CUDA_DEF DLIO_HAS_CUDA=0)
endif()
target_compile_definitions(nano_gicp PRIVATE ${_DLIO_CUDA_DEF})

# ---- Component Shared Libraries ----

# Odometry Component
add_library(dlio_odom_component SHARED
  src/dlio/odom/odom.cc
  src/dlio/odom/odom_callbacks.cc
  src/dlio/odom/odom_registration.cc
  src/dlio/odom/odom_keyframes.cc
  src/dlio/odom/odom_relocalization.cc
  src/dlio/odom/odom_services.cc
  src/dlio/odom/odom_submap_localization.cc
  src/dlio/odom/kfdb_io.cc
  src/dlio/odom/occupancy_grid.cc
  src/dlio/engines/registration_engine.cc
  src/dlio/engines/prefilter_engine.cc
  src/dlio/engines/appearance_engine.cc
  src/dlio/algorithms/robust_icp.cc
  src/dlio/algorithms/voxel_hash_map.cc
  src/dlio/algorithms/error_state_ekf.cc
)
target_compile_options(dlio_odom_component PRIVATE ${OpenMP_FLAGS})
target_compile_definitions(dlio_odom_component PRIVATE ${_DLIO_CUDA_DEF})
if(DLIO_HAS_CUDA)
  set(_odom_link_libs
    ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp dlio_cuda ${cpp_typesupport_target}
    g2o::core g2o::types_slam3d g2o::solver_eigen
    gtsam
    ${GeographicLib_LIBRARIES}
  )
  set(_odom_ament_deps
    rclcpp rclcpp_components tf2_ros pcl_ros nav_msgs pcl_conversions sensor_msgs geometry_msgs visualization_msgs ndt_omp_ros2
  )
  if(DLIO_HAS_NDT_CUDA)
    list(APPEND _odom_link_libs ${ndt_cuda_ros2_LIBRARIES})
    list(APPEND _odom_ament_deps ndt_cuda_ros2)
  endif()
  target_link_libraries(dlio_odom_component ${_odom_link_libs})
  ament_target_dependencies(dlio_odom_component ${_odom_ament_deps})
else()
  target_link_libraries(dlio_odom_component
    ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp ${cpp_typesupport_target}
    g2o::core g2o::types_slam3d g2o::solver_eigen
    gtsam
    ${GeographicLib_LIBRARIES}
  )
  ament_target_dependencies(dlio_odom_component
    rclcpp rclcpp_components tf2_ros pcl_ros nav_msgs pcl_conversions sensor_msgs geometry_msgs visualization_msgs ndt_omp_ros2
  )
endif()
rclcpp_components_register_nodes(dlio_odom_component "dlio::OdomNode")

# LIO-SAM Map Optimization Component (GTSAM iSAM2)
add_library(dlio_lio_sam_map_opt_component SHARED
  src/dlio/mapping/lio_sam_map_optimization.cc
  src/dlio/engines/registration_engine.cc
  src/dlio/engines/appearance_engine.cc
  src/dlio/algorithms/robust_icp.cc
)
target_compile_options(dlio_lio_sam_map_opt_component PRIVATE ${OpenMP_FLAGS})
target_compile_definitions(dlio_lio_sam_map_opt_component PRIVATE ${_DLIO_CUDA_DEF})
if(DLIO_HAS_CUDA)
  set(_mapopt_link_libs
    ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp dlio_cuda ${cpp_typesupport_target}
    gtsam
    ${GeographicLib_LIBRARIES}
  )
  set(_mapopt_ament_deps
    rclcpp rclcpp_components pcl_ros pcl_conversions nav_msgs sensor_msgs geometry_msgs visualization_msgs ndt_omp_ros2 tf2_ros
  )
  if(DLIO_HAS_NDT_CUDA)
    list(APPEND _mapopt_link_libs ${ndt_cuda_ros2_LIBRARIES})
    list(APPEND _mapopt_ament_deps ndt_cuda_ros2)
  endif()
  target_link_libraries(dlio_lio_sam_map_opt_component ${_mapopt_link_libs})
  ament_target_dependencies(dlio_lio_sam_map_opt_component ${_mapopt_ament_deps})
else()
  target_link_libraries(dlio_lio_sam_map_opt_component
    ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp ${cpp_typesupport_target}
    gtsam
    ${GeographicLib_LIBRARIES}
  )
  ament_target_dependencies(dlio_lio_sam_map_opt_component
    rclcpp rclcpp_components pcl_ros pcl_conversions nav_msgs sensor_msgs geometry_msgs visualization_msgs ndt_omp_ros2 tf2_ros
  )
endif()
rclcpp_components_register_nodes(dlio_lio_sam_map_opt_component "dlio::LioSamMapOptimizationNode")

# ---- Standalone Executables (backward compatibility) ----

add_executable(dlio_odom_node src/dlio/odom/odom_node.cc)
target_link_libraries(dlio_odom_node dlio_odom_component)
ament_target_dependencies(dlio_odom_node rclcpp)

add_executable(dlio_lio_sam_map_opt_node src/dlio/mapping/lio_sam_map_optimization_node.cc)
target_link_libraries(dlio_lio_sam_map_opt_node dlio_lio_sam_map_opt_component)
ament_target_dependencies(dlio_lio_sam_map_opt_node rclcpp)

# ---- Export & Install ----

ament_export_include_directories(include)
if(DLIO_HAS_CUDA)
  ament_export_libraries(dlio_odom_component dlio_lio_sam_map_opt_component nano_gicp nanoflann dlio_cuda)
else()
  ament_export_libraries(dlio_odom_component dlio_lio_sam_map_opt_component nano_gicp nanoflann)
endif()
ament_export_dependencies(rclcpp rclcpp_components std_msgs sensor_msgs geometry_msgs pcl_ros)
ament_package()

set(_install_lib_targets dlio_odom_component dlio_lio_sam_map_opt_component)
if(DLIO_HAS_CUDA)
  list(APPEND _install_lib_targets dlio_cuda)
endif()
install(TARGETS ${_install_lib_targets}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# ---- IMU Integrator (lightweight debug tool) ----
add_executable(imu_integrator_node src/dlio/tools/imu_integrator_node.cc)
target_include_directories(imu_integrator_node PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(imu_integrator_node rclcpp sensor_msgs nav_msgs geometry_msgs)

# ---- IMU-LiDAR Extrinsic Calibration Tool (pure GICP, no DLIO dependency) ----
add_executable(imu_lidar_calib_node src/dlio/tools/imu_lidar_calib_node.cc)
target_include_directories(imu_lidar_calib_node PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(imu_lidar_calib_node ${PCL_LIBRARIES})
ament_target_dependencies(imu_lidar_calib_node rclcpp sensor_msgs nav_msgs geometry_msgs pcl_conversions)

install(TARGETS
  dlio_odom_node dlio_lio_sam_map_opt_node imu_integrator_node imu_lidar_calib_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)
