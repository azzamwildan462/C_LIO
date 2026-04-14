#pragma once

#include "dlio/dlio.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <utility>

namespace dlio
{

  struct EkfParams
  {
    // IMU noise parameters (continuous-time spectral densities)
    double sigma_accel = 0.1;        // m/s^2/sqrt(Hz) — accel white noise
    double sigma_gyro = 0.01;        // rad/s/sqrt(Hz) — gyro white noise
    double sigma_accel_bias = 0.001; // m/s^3/sqrt(Hz) — accel random walk
    double sigma_gyro_bias = 0.0001; // rad/s^2/sqrt(Hz) — gyro random walk

    // Measurement noise base (realistic for GICP scan-to-submap)
    double sigma_pos_base = 0.1; // m
    double sigma_rot_base = 0.02; // rad

    // Uncertainty scaling weights
    double fitness_weight = 10.0;
    double space_weight = 0.5;
    double degeneracy_weight = 5.0; // scale sigma when scan geometry is degenerate (tunnel/corridor)

    // Uncertainty clamps
    double sigma_pos_min = 0.01;
    double sigma_pos_max = 5.0;
    double sigma_rot_min = 0.001;
    double sigma_rot_max = 0.5;

    // Mahalanobis gate: chi2(6,0.999)=22.46 — only reject true outliers
    double gate_threshold = 22.46;

    // Max fitness score to even attempt EKF update
    double max_fitness = 5.0;

    // Max consecutive rejections before auto-reset (prevents death spiral)
    int max_consecutive_rejects = 5;

    // Bias clamps (shared with geo observer)
    double abias_max = 5.0;
    double gbias_max = 0.5;

    // Initial covariance diagonal (conservative)
    double init_pos_std = 1.0;
    double init_vel_std = 1.0;
    double init_rot_std = 0.1;
    double init_abias_std = 0.5;
    double init_gbias_std = 0.1;
  };

  class ErrorStateEkf
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ErrorStateEkf() = default;
    void init(const EkfParams &params);

    /// Called per IMU sample — propagates covariance
    void predict(double dt,
                 const Eigen::Vector3f &accel_body,
                 const Eigen::Vector3f &gyro_body,
                 const Eigen::Quaternionf &q_nominal,
                 float gravity);

    /// Called per LiDAR scan — returns false if gated (dx_ = 0)
    bool update(const Eigen::Vector3f &z_pos,
                const Eigen::Quaternionf &z_quat,
                const Eigen::Vector3f &state_p,
                const Eigen::Quaternionf &state_q,
                double sigma_pos, double sigma_rot);

    const Eigen::Matrix<float, 15, 1> &getDeltaState() const { return dx_; }
    float lastMahalanobis() const { return last_mahal_; }
    int consecutiveRejects() const { return consecutive_rejects_; }
    const Eigen::Matrix<float, 15, 15> &getCovariance() const { return P_; }
    bool isInitialized() const { return initialized_; }

    /// Compute isotropic measurement noise from fitness + spaciousness + scatter
    static std::pair<double, double> computeMeasurementNoise(
        const EkfParams &params,
        double fitness_score,
        double spaciousness,
        double scatter);

    /// Compute scatter ratio (0-1) from global scan point distribution.
    /// 1.0 = well-distributed (good), 0.0 = degenerate (tunnel/corridor).
    static double computeScanScatter(
        const pcl::PointCloud<PointType>::ConstPtr &scan);

  private:
    static Eigen::Matrix3f skew(const Eigen::Vector3f &v);

    EkfParams params_;
    Eigen::Matrix<float, 15, 15> P_;
    Eigen::Matrix<float, 15, 1> dx_;
    bool initialized_ = false;
    int update_count_ = 0;
    int consecutive_rejects_ = 0;
    float last_mahal_ = 0.0f;
  };

} // namespace dlio
