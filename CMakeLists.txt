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

cmake_minimum_required(VERSION 3.12.4)
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
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)

# ---- Component Shared Libraries ----

# Odometry Component
add_library(dlio_odom_component SHARED
  src/dlio/odom.cc
  src/dlio/odom_callbacks.cc
  src/dlio/odom_registration.cc
  src/dlio/robust_icp.cc
  src/dlio/voxel_hash_map.cc
  src/dlio/registration_helper.cc
  src/dlio/odom_keyframes.cc
  src/dlio/odom_relocalization.cc
  src/dlio/odom_services.cc
  src/dlio/odom_submap_localization.cc
  src/dlio/kfdb_io.cc
  src/dlio/occupancy_grid.cc
)
target_compile_options(dlio_odom_component PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_odom_component
  ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp ${cpp_typesupport_target}
  g2o::core g2o::types_slam3d g2o::solver_eigen
  ${GeographicLib_LIBRARIES}
)
ament_target_dependencies(dlio_odom_component
  rclcpp rclcpp_components tf2_ros pcl_ros nav_msgs pcl_conversions sensor_msgs geometry_msgs ndt_omp_ros2
)
rclcpp_components_register_nodes(dlio_odom_component "dlio::OdomNode")

# Map Component
add_library(dlio_map_component SHARED
  src/dlio/map.cc
)
target_compile_options(dlio_map_component PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_map_component
  ${PCL_LIBRARIES} ${cpp_typesupport_target}
)
ament_target_dependencies(dlio_map_component
  rclcpp rclcpp_components tf2_ros pcl_ros pcl_conversions sensor_msgs geometry_msgs
)
rclcpp_components_register_nodes(dlio_map_component "dlio::MapNode")

# LIO-SAM Map Optimization Component (GTSAM iSAM2)
add_library(dlio_lio_sam_map_opt_component SHARED
  src/dlio/lio_sam_map_optimization.cc
  src/dlio/registration_helper.cc
  src/dlio/robust_icp.cc
)
target_compile_options(dlio_lio_sam_map_opt_component PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_lio_sam_map_opt_component
  ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp ${cpp_typesupport_target}
  gtsam
  ${GeographicLib_LIBRARIES}
)
ament_target_dependencies(dlio_lio_sam_map_opt_component
  rclcpp rclcpp_components pcl_ros pcl_conversions nav_msgs sensor_msgs geometry_msgs visualization_msgs ndt_omp_ros2 tf2_ros
)
rclcpp_components_register_nodes(dlio_lio_sam_map_opt_component "dlio::LioSamMapOptimizationNode")

# ---- Standalone Executables (backward compatibility) ----

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc)
target_link_libraries(dlio_odom_node dlio_odom_component)
ament_target_dependencies(dlio_odom_node rclcpp)

# Mapping Node
add_executable(dlio_map_node src/dlio/map_node.cc)
target_link_libraries(dlio_map_node dlio_map_component)
ament_target_dependencies(dlio_map_node rclcpp)

# Graph SLAM Node
# LIO-SAM Map Optimization Node
add_executable(dlio_lio_sam_map_opt_node src/dlio/lio_sam_map_optimization_node.cc)
target_link_libraries(dlio_lio_sam_map_opt_node dlio_lio_sam_map_opt_component)
ament_target_dependencies(dlio_lio_sam_map_opt_node rclcpp)

# ---- Export & Install ----

ament_export_include_directories(include)
ament_export_libraries(dlio_odom_component dlio_map_component dlio_lio_sam_map_opt_component nano_gicp nanoflann)
ament_export_dependencies(rclcpp rclcpp_components std_msgs sensor_msgs geometry_msgs pcl_ros)
ament_package()

install(TARGETS
  dlio_odom_component
  dlio_map_component
  dlio_lio_sam_map_opt_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  dlio_odom_node dlio_map_node dlio_lio_sam_map_opt_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)
