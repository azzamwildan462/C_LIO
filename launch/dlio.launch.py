#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Set default arguments
    rviz = LaunchConfiguration('rviz', default='false')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='points_raw')
    imu_topic = LaunchConfiguration('imu_topic', default='imu_raw')
    map_mode = LaunchConfiguration('map_mode', default='localization')
    map_path = LaunchConfiguration('map_path', default='')
    relocalize = LaunchConfiguration('relocalize', default='true')
    use_corrected = LaunchConfiguration('use_corrected', default='true')

    # Define arguments
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value=rviz,
        description='Launch RViz'
    )
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value=pointcloud_topic,
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value=imu_topic,
        description='IMU topic name'
    )
    declare_map_mode_arg = DeclareLaunchArgument(
        'map_mode',
        default_value=map_mode,
        description='Map mode: mapping or localization'
    )
    declare_map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value=map_path,
        description='Path to PCD map file (load/save)'
    )
    declare_relocalize_arg = DeclareLaunchArgument(
        'relocalize',
        default_value=relocalize,
        description='Enable Scan Context relocalization for initial pose'
    )
    declare_use_corrected_arg = DeclareLaunchArgument(
        'use_corrected',
        default_value=use_corrected,
        description='Load graph-optimized corrected map files in localization mode'
    )

    # Load parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])
    graph_slam_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'graph_slam.yaml'])

    # Map params override (passed to both OdomNode and MapNode)
    map_params = {'map/mode': map_mode, 'map/path': map_path, 'map/use_corrected': use_corrected}
    odom_extra_params = {'map/relocalize': relocalize}

    # Composable Node Container (all components in one process with IPC)
    dlio_container = ComposableNodeContainer(
        name='dlio_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # DLIO Odometry Component
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::OdomNode',
                name='dlio_odom',
                parameters=[dlio_yaml_path, dlio_params_yaml_path, map_params, odom_extra_params],
                remappings=[
                    ('pointcloud', pointcloud_topic),
                    ('imu', imu_topic),
                    ('odom', 'dlio/odom_node/odom'),
                    ('pose', 'dlio/odom_node/pose'),
                    ('path', 'dlio/odom_node/path'),
                    ('kf_pose', 'dlio/odom_node/keyframes'),
                    ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
                    ('kf_stamped', 'dlio/odom_node/keyframe_stamped'),
                    ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
                    ('dlio_odom/set_mode', 'dlio/odom_node/set_mode'),
                    ('dlio_odom/relocalize', 'dlio/odom_node/relocalize'),
                    ('dlio_odom/set_pose', 'dlio/odom_node/set_pose'),
                    ('dlio_odom/get_state', 'dlio/odom_node/get_state'),
                    ('dlio_odom/new_map', 'dlio/odom_node/new_map'),
                    ('dlio_odom/new_map_w_zero', 'dlio/odom_node/new_map_w_zero'),
                    ('save_pcd_map', 'dlio/map_node/save_pcd'),
                    ('save_corrected_pcd', 'dlio/graph_slam/save_corrected_pcd'),
                    ('corrected_kf_poses', 'dlio/graph_slam/corrected_kf_poses'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # DLIO Mapping Component (IPC disabled — kf_cloud published from detached thread)
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::MapNode',
                name='dlio_map',
                parameters=[dlio_yaml_path, dlio_params_yaml_path, map_params],
                remappings=[
                    ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
                    ('map', 'dlio/map_node/map'),
                    ('save_pcd', 'dlio/map_node/save_pcd'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # DLIO Graph SLAM Component
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::GraphSlamNode',
                name='dlio_graph_slam',
                parameters=[dlio_yaml_path, dlio_params_yaml_path, graph_slam_yaml_path, map_params],
                remappings=[
                    ('keyframe_stamped', 'dlio/odom_node/keyframe_stamped'),
                    ('corrected_path', 'dlio/graph_slam/corrected_path'),
                    ('corrected_map', 'dlio/graph_slam/corrected_map'),
                    ('corrected_kf_poses', 'dlio/graph_slam/corrected_kf_poses'),
                    ('loop_closures', 'dlio/graph_slam/loop_closures'),
                    ('save_corrected_pcd', 'dlio/graph_slam/save_corrected_pcd'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    # RViz node
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'dlio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_map_mode_arg,
        declare_map_path_arg,
        declare_relocalize_arg,
        declare_use_corrected_arg,
        dlio_container,
        rviz_node
    ])
