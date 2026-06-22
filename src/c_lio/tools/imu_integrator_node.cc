#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <cmath>

class ImuIntegratorNode : public rclcpp::Node
{
public:
  ImuIntegratorNode() : Node("imu_integrator")
  {
    // Parameters
    this->declare_parameter("calibration_time", 3.0);
    this->declare_parameter("odom/gravity", 9.80665);
    this->declare_parameter("odom/imu/gravityRemoved", false);
    this->declare_parameter("publish_rate", 10.0);
    this->declare_parameter("frame_id", "odom");

    calibration_time_ = this->get_parameter("calibration_time").as_double();
    gravity_ = static_cast<float>(this->get_parameter("odom/gravity").as_double());
    gravity_removed_ = this->get_parameter("odom/imu/gravityRemoved").as_bool();
    publish_rate_ = this->get_parameter("publish_rate").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();

    if (gravity_removed_)
      RCLCPP_INFO(this->get_logger(), "IMU gravity-removed mode: adding +g to Z");

    // Extrinsics: baselink2imu (same params as C_LIO odom)
    loadExtrinsics();

    // State
    position_ = Eigen::Vector3f::Zero();
    velocity_ = Eigen::Vector3f::Zero();
    orientation_ = Eigen::Quaternionf::Identity();
    gyro_bias_ = Eigen::Vector3f::Zero();
    calibrated_ = false;
    prev_stamp_ = 0.0;
    prev_ang_vel_cg_ = Eigen::Vector3f::Zero();

    // Path
    path_msg_.header.frame_id = frame_id_;

    // Publishers
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("imu_path", 1);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("imu_odom", 1);

    // Subscriber (topic "imu" — remapped via launch file, same as C_LIO odom)
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu", rclcpp::SensorDataQoS(),
        std::bind(&ImuIntegratorNode::callbackImu, this, std::placeholders::_1));

    // Publish timer
    double period = 1.0 / publish_rate_;
    publish_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(period),
        std::bind(&ImuIntegratorNode::publishPath, this));

    RCLCPP_INFO(this->get_logger(), "IMU Integrator started (calib=%.1fs, g=%.4f)",
                calibration_time_, gravity_);
  }

private:
  void loadExtrinsics()
  {
    // Defaults
    std::vector<double> t_default = {0.0, 0.0, 0.0};
    std::vector<double> rpy_default = {0.0, 0.0, 0.0};
    std::vector<double> R_default = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    this->declare_parameter("extrinsics/baselink2imu/t", t_default);
    this->declare_parameter("extrinsics/baselink2imu/rpy", rpy_default);
    this->declare_parameter("extrinsics/baselink2imu/R", R_default);

    auto t_vec = this->get_parameter("extrinsics/baselink2imu/t").as_double_array();
    auto rpy_vec = this->get_parameter("extrinsics/baselink2imu/rpy").as_double_array();
    auto R_vec = this->get_parameter("extrinsics/baselink2imu/R").as_double_array();

    ext_t_ = Eigen::Vector3f(t_vec[0], t_vec[1], t_vec[2]);

    if (rpy_vec != rpy_default)
    {
      // RPY in degrees → rotation matrix (ZYX order, same as C_LIO)
      float r = static_cast<float>(rpy_vec[0]) * M_PI / 180.f;
      float p = static_cast<float>(rpy_vec[1]) * M_PI / 180.f;
      float y = static_cast<float>(rpy_vec[2]) * M_PI / 180.f;
      ext_R_ = Eigen::AngleAxisf(y, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(p, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(r, Eigen::Vector3f::UnitX());
      RCLCPP_INFO(this->get_logger(), "baselink2imu: rpy=[%.1f, %.1f, %.1f] deg",
                  rpy_vec[0], rpy_vec[1], rpy_vec[2]);
    }
    else
    {
      // Fallback to rotation matrix
      std::vector<float> Rf(R_vec.begin(), R_vec.end());
      ext_R_ = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(Rf.data());
    }

    bool is_identity = ext_R_.isApprox(Eigen::Matrix3f::Identity(), 1e-4f) &&
                       ext_t_.isApprox(Eigen::Vector3f::Zero(), 1e-4f);
    if (!is_identity)
    {
      RCLCPP_INFO(this->get_logger(), "IMU extrinsics loaded: t=[%.3f, %.3f, %.3f]",
                  ext_t_[0], ext_t_[1], ext_t_[2]);
    }
  }

  void transformImu(const Eigen::Vector3f &ang_vel_raw, const Eigen::Vector3f &lin_accel_raw,
                    float dt, Eigen::Vector3f &ang_vel_out, Eigen::Vector3f &lin_accel_out)
  {
    // Rotate angular velocity to baselink frame
    ang_vel_out = ext_R_ * ang_vel_raw;

    // Rotate linear acceleration + compensate for lever arm
    lin_accel_out = ext_R_ * lin_accel_raw;
    if (!ext_t_.isApprox(Eigen::Vector3f::Zero(), 1e-6f) && dt > 0.f)
    {
      Eigen::Vector3f ang_accel = (ang_vel_out - prev_ang_vel_cg_) / dt;
      lin_accel_out += ang_accel.cross(-ext_t_) + ang_vel_out.cross(ang_vel_out.cross(-ext_t_));
    }
    prev_ang_vel_cg_ = ang_vel_out;
  }

  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    double stamp = rclcpp::Time(msg->header.stamp).seconds();

    Eigen::Vector3f ang_vel_raw(
        static_cast<float>(msg->angular_velocity.x),
        static_cast<float>(msg->angular_velocity.y),
        static_cast<float>(msg->angular_velocity.z));
    Eigen::Vector3f lin_accel_raw(
        static_cast<float>(msg->linear_acceleration.x),
        static_cast<float>(msg->linear_acceleration.y),
        static_cast<float>(msg->linear_acceleration.z));

    // Compute dt for extrinsic transform
    float dt_raw = (prev_raw_stamp_ > 0.0) ? static_cast<float>(stamp - prev_raw_stamp_) : 0.005f;
    prev_raw_stamp_ = stamp;

    // If IMU driver already removed gravity, add it back (same as C_LIO odom)
    if (gravity_removed_)
    {
      lin_accel_raw[2] += gravity_;
    }

    // Transform IMU to baselink frame (same as C_LIO odom)
    Eigen::Vector3f ang_vel, lin_accel;
    transformImu(ang_vel_raw, lin_accel_raw, dt_raw, ang_vel, lin_accel);

    // --- Calibration phase ---
    if (!calibrated_)
    {
      if (calib_samples_.empty())
      {
        calib_start_time_ = stamp;
        RCLCPP_INFO(this->get_logger(), "Calibrating IMU (%.1fs)...", calibration_time_);
      }

      calib_gyro_sum_ += ang_vel;
      calib_accel_sum_ += lin_accel;
      calib_samples_.push_back(stamp);

      if (stamp - calib_start_time_ >= calibration_time_)
      {
        int n = static_cast<int>(calib_samples_.size());
        gyro_bias_ = calib_gyro_sum_ / static_cast<float>(n);
        Eigen::Vector3f accel_mean = calib_accel_sum_ / static_cast<float>(n);

        // Gravity alignment: align mean accel to [0, 0, g]
        Eigen::Vector3f grav_est = accel_mean.normalized() * gravity_;
        orientation_ = Eigen::Quaternionf::FromTwoVectors(
            grav_est, Eigen::Vector3f(0.f, 0.f, gravity_));

        calibrated_ = true;
        prev_stamp_ = stamp;

        RCLCPP_INFO(this->get_logger(),
                    "Calibration done (%d samples). gyro_bias=[%.5f, %.5f, %.5f] gravity_dir=[%.3f, %.3f, %.3f]",
                    n, gyro_bias_[0], gyro_bias_[1], gyro_bias_[2],
                    accel_mean[0], accel_mean[1], accel_mean[2]);
      }
      return;
    }

    // --- Integration phase ---
    float dt = static_cast<float>(stamp - prev_stamp_);
    prev_stamp_ = stamp;

    if (dt <= 0.f || dt > 0.5f)
      return;

    // Subtract gyro bias
    Eigen::Vector3f w = ang_vel - gyro_bias_;

    // Quaternion integration (half-angle)
    Eigen::Quaternionf dq;
    float half_dt = 0.5f * dt;
    dq.w() = 1.0f;
    dq.x() = w[0] * half_dt;
    dq.y() = w[1] * half_dt;
    dq.z() = w[2] * half_dt;
    dq.normalize();
    orientation_ = (orientation_ * dq).normalized();

    // Rotate accel to world frame
    Eigen::Vector3f world_accel = orientation_._transformVector(lin_accel);

    // Subtract gravity
    world_accel[2] -= gravity_;

    // Double integration
    position_ += velocity_ * dt + 0.5f * world_accel * dt * dt;
    velocity_ += world_accel * dt;

    // Store latest for publishing
    latest_stamp_ = msg->header.stamp;
    has_new_data_ = true;
  }

  void publishPath()
  {
    if (!calibrated_ || !has_new_data_)
      return;
    has_new_data_ = false;

    // Append to path
    geometry_msgs::msg::PoseStamped p;
    p.header.stamp = latest_stamp_;
    p.header.frame_id = frame_id_;
    p.pose.position.x = position_[0];
    p.pose.position.y = position_[1];
    p.pose.position.z = position_[2];
    p.pose.orientation.w = orientation_.w();
    p.pose.orientation.x = orientation_.x();
    p.pose.orientation.y = orientation_.y();
    p.pose.orientation.z = orientation_.z();

    path_msg_.header.stamp = latest_stamp_;
    path_msg_.poses.push_back(p);
    path_pub_->publish(path_msg_);

    // Also publish odometry
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = latest_stamp_;
    odom.header.frame_id = frame_id_;
    odom.child_frame_id = "imu_link";
    odom.pose.pose = p.pose;
    odom.twist.twist.linear.x = velocity_[0];
    odom.twist.twist.linear.y = velocity_[1];
    odom.twist.twist.linear.z = velocity_[2];
    odom_pub_->publish(odom);
  }

  // Parameters
  std::string frame_id_;
  double calibration_time_;
  float gravity_;
  bool gravity_removed_ = false;
  double publish_rate_;

  // Extrinsics: baselink2imu
  Eigen::Matrix3f ext_R_ = Eigen::Matrix3f::Identity();
  Eigen::Vector3f ext_t_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f prev_ang_vel_cg_ = Eigen::Vector3f::Zero();
  double prev_raw_stamp_ = 0.0;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // State
  Eigen::Vector3f position_;
  Eigen::Vector3f velocity_;
  Eigen::Quaternionf orientation_;
  Eigen::Vector3f gyro_bias_;
  bool calibrated_ = false;
  double prev_stamp_ = 0.0;
  builtin_interfaces::msg::Time latest_stamp_;
  bool has_new_data_ = false;

  // Calibration accumulators
  std::vector<double> calib_samples_;
  Eigen::Vector3f calib_gyro_sum_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f calib_accel_sum_ = Eigen::Vector3f::Zero();
  double calib_start_time_ = 0.0;

  // Path message
  nav_msgs::msg::Path path_msg_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuIntegratorNode>());
  rclcpp::shutdown();
  return 0;
}
