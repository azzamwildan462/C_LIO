#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <deque>
#include <cmath>

using PointType = pcl::PointXYZ;

class ImuLidarCalibNode : public rclcpp::Node
{
public:
  ImuLidarCalibNode() : Node("imu_lidar_calib")
  {
    this->declare_parameter("num_samples", 300);
    this->declare_parameter("min_rotation_deg", 1.0);
    this->declare_parameter("voxel_size", 0.5);
    this->declare_parameter("max_icp_dist", 5.0);
    num_samples_ = this->get_parameter("num_samples").as_int();
    min_rotation_deg_ = this->get_parameter("min_rotation_deg").as_double();
    voxel_size_ = this->get_parameter("voxel_size").as_double();
    max_icp_dist_ = this->get_parameter("max_icp_dist").as_double();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu", rclcpp::SensorDataQoS(),
        std::bind(&ImuLidarCalibNode::callbackImu, this, std::placeholders::_1));

    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud", rclcpp::SensorDataQoS(),
        std::bind(&ImuLidarCalibNode::callbackPointcloud, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "IMU-LiDAR Extrinsic Calibration (pure ICP, %d samples, voxel=%.2f, min_rot=%.1f deg)",
                num_samples_, voxel_size_, min_rotation_deg_);
  }

private:
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x,
                         msg->orientation.y, msg->orientation.z);
    if (q.squaredNorm() < 0.1)
      return;
    double stamp = rclcpp::Time(msg->header.stamp).seconds();
    std::lock_guard<std::mutex> lock(imu_mtx_);
    imu_buffer_.push_back({stamp, q.normalized()});
    if (imu_buffer_.size() > 10000)
      imu_buffer_.pop_front();
  }

  void callbackPointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (solved_)
      return;

    // Convert + downsample
    auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty())
      return;

    auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::VoxelGrid<PointType> vf;
    vf.setInputCloud(cloud);
    vf.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    vf.filter(*filtered);

    if (filtered->size() < 500)
      return;

    double stamp = rclcpp::Time(msg->header.stamp).seconds();

    // Get IMU orientation at this timestamp
    Eigen::Quaterniond q_imu;
    if (!getImuAtTime(stamp, q_imu))
    {
      std::lock_guard<std::mutex> lock(imu_mtx_);
      if (!imu_buffer_.empty())
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                             "IMU time mismatch: pc=%.3f imu=[%.3f..%.3f] diff=%.3fs",
                             stamp, imu_buffer_.front().stamp, imu_buffer_.back().stamp,
                             std::abs(stamp - imu_buffer_.back().stamp));
      }
      else
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "IMU buffer empty");
      }
      return;
    }

    if (!prev_cloud_)
    {
      prev_cloud_ = filtered;
      prev_stamp_ = stamp;
      prev_q_imu_ = q_imu;
      RCLCPP_INFO(this->get_logger(), "First scan received (%zu pts). Waiting for next...",
                  filtered->size());
      return;
    }

    // Run GICP: pure LiDAR scan matching (no IMU initial guess)
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
    gicp.setInputSource(filtered);
    gicp.setInputTarget(prev_cloud_);
    gicp.setMaxCorrespondenceDistance(max_icp_dist_);
    gicp.setMaximumIterations(50);
    gicp.setTransformationEpsilon(1e-6);

    pcl::PointCloud<PointType> aligned;
    gicp.align(aligned);

    bool converged = gicp.hasConverged();
    double fitness = gicp.getFitnessScore();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                         "GICP: converged=%d fitness=%.4f src=%zu tgt=%zu",
                         converged, fitness, filtered->size(), prev_cloud_->size());

    if (!converged || fitness > 2.0)
    {
      prev_cloud_ = filtered;
      prev_stamp_ = stamp;
      prev_q_imu_ = q_imu;
      return;
    }

    // Extract pure LiDAR delta rotation
    Eigen::Matrix4f T = gicp.getFinalTransformation();
    Eigen::Matrix3d delta_lidar = T.block<3, 3>(0, 0).cast<double>();

    // IMU delta rotation
    Eigen::Quaterniond delta_imu_q = prev_q_imu_.conjugate() * q_imu;
    delta_imu_q.normalize();
    Eigen::Matrix3d delta_imu = delta_imu_q.toRotationMatrix();

    // Skip if rotation too small
    double angle_deg = 2.0 * std::acos(std::min(std::abs(delta_imu_q.w()), 1.0)) * 180.0 / M_PI;

    prev_cloud_ = filtered;
    prev_stamp_ = stamp;
    prev_q_imu_ = q_imu;

    if (angle_deg < min_rotation_deg_)
      return;

    // Store pair
    delta_lidar_list_.push_back(delta_lidar);
    delta_imu_list_.push_back(delta_imu);

    int n = static_cast<int>(delta_lidar_list_.size());
    if (n % 25 == 0)
    {
      RCLCPP_INFO(this->get_logger(), "Collected %d / %d pairs (last angle=%.2f deg, fitness=%.4f)",
                  n, num_samples_, angle_deg, gicp.getFitnessScore());
    }

    if (n >= num_samples_)
      solve();
  }

  bool getImuAtTime(double timestamp, Eigen::Quaterniond &out)
  {
    std::lock_guard<std::mutex> lock(imu_mtx_);
    if (imu_buffer_.empty())
      return false;
    double best_dt = 1e9;
    int best_idx = -1;
    for (int i = 0; i < static_cast<int>(imu_buffer_.size()); i++)
    {
      double dt = std::abs(imu_buffer_[i].stamp - timestamp);
      if (dt < best_dt)
      {
        best_dt = dt;
        best_idx = i;
      }
    }
    if (best_idx < 0 || best_dt > 1.0)
      return false;
    out = imu_buffer_[best_idx].q;
    return true;
  }

  void solve()
  {
    solved_ = true;
    int n = static_cast<int>(delta_lidar_list_.size());
    RCLCPP_INFO(this->get_logger(), "Solving extrinsic rotation from %d pure-ICP pairs...", n);

    // SVD solution: find R such that R × delta_imu_i ≈ delta_lidar_i × R
    // Approximate: H = sum(delta_lidar_i × delta_imu_i^T), R = U × V^T from SVD(H)
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n; i++)
      H += delta_lidar_list_[i] * delta_imu_list_[i].transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_opt = U * V.transpose();
    if (R_opt.determinant() < 0)
    {
      V.col(2) *= -1;
      R_opt = U * V.transpose();
    }

    // Extract RPY (ZYX)
    double roll = std::atan2(R_opt(2, 1), R_opt(2, 2));
    double pitch = std::asin(std::clamp(-R_opt(2, 0), -1.0, 1.0));
    double yaw = std::atan2(R_opt(1, 0), R_opt(0, 0));
    double r_deg = roll * 180.0 / M_PI;
    double p_deg = pitch * 180.0 / M_PI;
    double y_deg = yaw * 180.0 / M_PI;

    // Residual
    double total_err = 0.0;
    for (int i = 0; i < n; i++)
    {
      Eigen::Matrix3d predicted = R_opt * delta_imu_list_[i] * R_opt.transpose();
      Eigen::Matrix3d err = delta_lidar_list_[i].transpose() * predicted;
      double trace = std::clamp((err.trace() - 1.0) / 2.0, -1.0, 1.0);
      double angle = std::acos(trace);
      total_err += angle * angle;
    }
    double rmse_deg = std::sqrt(total_err / n) * 180.0 / M_PI;

    printf("\n");
    printf("============================================================\n");
    printf("  IMU-LiDAR Extrinsic Rotation Calibration (Pure ICP)\n");
    printf("============================================================\n");
    printf("  Pairs used:    %d\n", n);
    printf("  RMSE residual: %.4f degrees\n", rmse_deg);
    printf("\n");
    printf("  Optimal baselink2imu RPY (degrees):\n");
    printf("    roll:  %10.4f\n", r_deg);
    printf("    pitch: %10.4f\n", p_deg);
    printf("    yaw:   %10.4f\n", y_deg);
    printf("\n");
    printf("  Copy to sensor.yaml:\n");
    printf("    extrinsics/baselink2imu/rpy: [%.4f, %.4f, %.4f]\n", r_deg, p_deg, y_deg);
    printf("============================================================\n");
    fflush(stdout);
  }

  struct ImuSample
  {
    double stamp;
    Eigen::Quaterniond q;
  };
  std::deque<ImuSample> imu_buffer_;
  std::mutex imu_mtx_;

  pcl::PointCloud<PointType>::Ptr prev_cloud_;
  double prev_stamp_ = 0.0;
  Eigen::Quaterniond prev_q_imu_;

  std::vector<Eigen::Matrix3d> delta_lidar_list_;
  std::vector<Eigen::Matrix3d> delta_imu_list_;

  int num_samples_;
  double min_rotation_deg_;
  double voxel_size_;
  double max_icp_dist_;
  bool solved_ = false;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuLidarCalibNode>());
  rclcpp::shutdown();
  return 0;
}
