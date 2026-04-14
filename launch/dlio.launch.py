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
    gps_topic = LaunchConfiguration('gps_topic', default='gps_raw')
    ext_odom_topic = LaunchConfiguration('ext_odom_topic', default='/odom')
    map_mode = LaunchConfiguration('map_mode', default='mapping')
    map_path = LaunchConfiguration('map_path', default='')
    relocalize = LaunchConfiguration('relocalize', default='false')
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
    declare_gps_topic_arg = DeclareLaunchArgument(
        'gps_topic',
        default_value=gps_topic,
        description='gps topic name'
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
    declare_ext_odom_topic_arg = DeclareLaunchArgument(
        'ext_odom_topic',
        default_value=ext_odom_topic,
        description='External odometry topic (wheel encoder / visual odom)'
    )
    declare_use_corrected_arg = DeclareLaunchArgument(
        'use_corrected',
        default_value=use_corrected,
        description='Load graph-optimized corrected map files in localization mode'
    )

    # Load parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    sensor_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'sensor.yaml'])
    odom_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'odom.yaml'])
    fusion_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'fusion.yaml'])
    map_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'map.yaml'])
    occupancy_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'occupancy_grid.yaml'])
    lio_sam_opt_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'lio_sam_map_optimization.yaml'])

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
                parameters=[dlio_yaml_path, sensor_yaml_path, odom_yaml_path, fusion_yaml_path, map_yaml_path, occupancy_yaml_path, map_params, odom_extra_params, {'gps/topic': gps_topic, 'odom/external_odom/topic': ext_odom_topic}],
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
                    ('deskewed_raw', 'dlio/odom_node/pointcloud/deskewed_raw'),
                    ('dlio_odom/set_mode', 'dlio/odom_node/set_mode'),
                    ('dlio_odom/relocalize', 'dlio/odom_node/relocalize'),
                    ('dlio_odom/set_pose', 'dlio/odom_node/set_pose'),
                    ('dlio_odom/get_state', 'dlio/odom_node/get_state'),
                    ('dlio_odom/new_map', 'dlio/odom_node/new_map'),
                    ('dlio_odom/new_map_w_zero', 'dlio/odom_node/new_map_w_zero'),
                    ('save_pcd_map', 'dlio/map_node/save_pcd'),
                    ('save_corrected_pcd', 'dlio/lio_sam_opt/save_corrected_pcd'),
                    ('corrected_kf_poses', 'dlio/lio_sam_opt/corrected_kf_poses'),
                    ('occupancy_grid', 'dlio/odom_node/occupancy_grid'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # LIO-SAM Map Optimization runs in separate process (see lio_sam_opt_container below)
            # to prevent heavy ICP/GTSAM work from starving OdomNode threads.
        ],
        output='screen',
    )

    # LIO-SAM Map Optimization — separate process with lower priority to avoid starving OdomNode
    lio_sam_opt_container = ComposableNodeContainer(
        name='lio_sam_opt_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        prefix='nice -n 10',
        composable_node_descriptions=[
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::LioSamMapOptimizationNode',
                name='dlio_lio_sam_map_opt',
                parameters=[dlio_yaml_path, sensor_yaml_path, map_yaml_path, lio_sam_opt_yaml_path, map_params, {'gps/topic': gps_topic}],
                remappings=[
                    ('keyframe_stamped', 'dlio/odom_node/keyframe_stamped'),
                    ('corrected_path', 'dlio/lio_sam_opt/corrected_path'),
                    ('corrected_map', 'dlio/lio_sam_opt/corrected_map'),
                    ('corrected_kf_poses', 'dlio/lio_sam_opt/corrected_kf_poses'),
                    ('loop_closures', 'dlio/lio_sam_opt/loop_closures'),
                    ('save_corrected_pcd', 'dlio/lio_sam_opt/save_corrected_pcd'),
                ],
            ),
        ],
        output='screen',
    )

    # IMU Integrator (debug: pure IMU trajectory)
    imu_integrator_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='imu_integrator_node',
        name='imu_integrator',
        parameters=[dlio_yaml_path, sensor_yaml_path],
        remappings=[
            ('imu', imu_topic),
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
        declare_gps_topic_arg,
        declare_ext_odom_topic_arg,
        declare_map_mode_arg,
        declare_map_path_arg,
        declare_relocalize_arg,
        declare_use_corrected_arg,
        dlio_container,
        lio_sam_opt_container,
        # imu_integrator_node,
        rviz_node
    ])
