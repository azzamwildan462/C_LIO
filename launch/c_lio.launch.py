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
    current_pkg = FindPackageShare('c_lio')

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
    c_lio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'c_lio.yaml'])
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
    c_lio_container = ComposableNodeContainer(
        name='c_lio_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # C_LIO Odometry Component
            ComposableNode(
                package='c_lio',
                plugin='c_lio::OdomNode',
                name='c_lio_odom',
                parameters=[c_lio_yaml_path, sensor_yaml_path, odom_yaml_path, fusion_yaml_path, map_yaml_path, occupancy_yaml_path, map_params, odom_extra_params, {'gps/topic': gps_topic, 'odom/external_odom/topic': ext_odom_topic}],
                remappings=[
                    ('pointcloud', pointcloud_topic),
                    ('imu', imu_topic),
                    ('odom', 'c_lio/odom_node/odom'),
                    ('pose', 'c_lio/odom_node/pose'),
                    ('path', 'c_lio/odom_node/path'),
                    ('kf_pose', 'c_lio/odom_node/keyframes'),
                    ('kf_cloud', 'c_lio/odom_node/pointcloud/keyframe'),
                    ('kf_stamped', 'c_lio/odom_node/keyframe_stamped'),
                    ('deskewed', 'c_lio/odom_node/pointcloud/deskewed'),
                    ('deskewed_raw', 'c_lio/odom_node/pointcloud/deskewed_raw'),
                    ('c_lio_odom/set_mode', 'c_lio/odom_node/set_mode'),
                    ('c_lio_odom/relocalize', 'c_lio/odom_node/relocalize'),
                    ('c_lio_odom/set_pose', 'c_lio/odom_node/set_pose'),
                    ('c_lio_odom/get_state', 'c_lio/odom_node/get_state'),
                    ('c_lio_odom/new_map', 'c_lio/odom_node/new_map'),
                    ('c_lio_odom/new_map_w_zero', 'c_lio/odom_node/new_map_w_zero'),
                    ('save_pcd_map', 'c_lio/map_node/save_pcd'),
                    ('save_corrected_pcd', 'c_lio/lio_sam_opt/save_corrected_pcd'),
                    ('corrected_kf_poses', 'c_lio/lio_sam_opt/corrected_kf_poses'),
                    ('occupancy_grid', 'c_lio/odom_node/occupancy_grid'),
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
                package='c_lio',
                plugin='c_lio::LioSamMapOptimizationNode',
                name='c_lio_lio_sam_map_opt',
                parameters=[c_lio_yaml_path, sensor_yaml_path, map_yaml_path, lio_sam_opt_yaml_path, map_params, {'gps/topic': gps_topic}],
                remappings=[
                    ('keyframe_stamped', 'c_lio/odom_node/keyframe_stamped'),
                    ('corrected_path', 'c_lio/lio_sam_opt/corrected_path'),
                    ('corrected_map', 'c_lio/lio_sam_opt/corrected_map'),
                    ('corrected_kf_poses', 'c_lio/lio_sam_opt/corrected_kf_poses'),
                    ('loop_closures', 'c_lio/lio_sam_opt/loop_closures'),
                    ('save_corrected_pcd', 'c_lio/lio_sam_opt/save_corrected_pcd'),
                    ('odom', 'c_lio/odom_node/odom'),
                    ('corrected_fusion_path', 'c_lio/lio_sam_opt/corrected_fusion_path'),
                    ('corrected_fusion_odom', 'c_lio/lio_sam_opt/corrected_fusion_odom'),
                ],
            ),
        ],
        output='screen',
    )

    # IMU Integrator (debug: pure IMU trajectory)
    imu_integrator_node = Node(
        package='c_lio',
        executable='imu_integrator_node',
        name='imu_integrator',
        parameters=[c_lio_yaml_path, sensor_yaml_path],
        remappings=[
            ('imu', imu_topic),
        ],
        output='screen',
    )

    # RViz node
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'c_lio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='c_lio_rviz',
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
        c_lio_container,
        lio_sam_opt_container,
        # imu_integrator_node,
        rviz_node
    ])
