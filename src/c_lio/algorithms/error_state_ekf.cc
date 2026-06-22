#include "c_lio/algorithms/error_state_ekf.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace c_lio
{

  Eigen::Matrix3f ErrorStateEkf::skew(const Eigen::Vector3f &v)
  {
    Eigen::Matrix3f S;
    S << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return S;
  }

  void ErrorStateEkf::init(const EkfParams &params)
  {
    params_ = params;
    dx_.setZero();

    P_.setZero();
    float p_pos = params_.init_pos_std * params_.init_pos_std;
    float p_vel = params_.init_vel_std * params_.init_vel_std;
    float p_rot = params_.init_rot_std * params_.init_rot_std;
    float p_ba = params_.init_abias_std * params_.init_abias_std;
    float p_bg = params_.init_gbias_std * params_.init_gbias_std;

    P_.diagonal() << p_pos, p_pos, p_pos,
        p_vel, p_vel, p_vel,
        p_rot, p_rot, p_rot,
        p_ba, p_ba, p_ba,
        p_bg, p_bg, p_bg;

    update_count_ = 0;
    consecutive_rejects_ = 0;
    initialized_ = true;
  }

  void ErrorStateEkf::predict(double dt,
                              const Eigen::Vector3f &accel_body,
                              const Eigen::Vector3f &gyro_body,
                              const Eigen::Quaternionf &q_nominal,
                              float gravity)
  {
    if (!initialized_ || dt <= 0 || dt > 1.0)
      return;

    float dtf = static_cast<float>(dt);
    Eigen::Matrix3f R = q_nominal.toRotationMatrix();

    Eigen::Matrix<float, 15, 15> F = Eigen::Matrix<float, 15, 15>::Identity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dtf;
    F.block<3, 3>(3, 6) = -R * skew(accel_body) * dtf;
    F.block<3, 3>(3, 9) = -R * dtf;
    F.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() - skew(gyro_body) * dtf;
    F.block<3, 3>(6, 12) = -Eigen::Matrix3f::Identity() * dtf;

    Eigen::Matrix<float, 15, 12> G = Eigen::Matrix<float, 15, 12>::Zero();
    G.block<3, 3>(3, 0) = -R;
    G.block<3, 3>(6, 3) = -Eigen::Matrix3f::Identity();
    G.block<3, 3>(9, 6) = Eigen::Matrix3f::Identity();
    G.block<3, 3>(12, 9) = Eigen::Matrix3f::Identity();

    float sa2 = params_.sigma_accel * params_.sigma_accel;
    float sg2 = params_.sigma_gyro * params_.sigma_gyro;
    float sba2 = params_.sigma_accel_bias * params_.sigma_accel_bias;
    float sbg2 = params_.sigma_gyro_bias * params_.sigma_gyro_bias;

    Eigen::Matrix<float, 12, 12> Qi = Eigen::Matrix<float, 12, 12>::Zero();
    Qi.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * sa2;
    Qi.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * sg2;
    Qi.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * sba2;
    Qi.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * sbg2;

    P_ = F * P_ * F.transpose() + G * Qi * G.transpose() * dtf;

    if (++update_count_ % 100 == 0)
    {
      P_ = 0.5f * (P_ + P_.transpose());
    }
  }

  bool ErrorStateEkf::update(const Eigen::Vector3f &z_pos,
                             const Eigen::Quaternionf &z_quat,
                             const Eigen::Vector3f &state_p,
                             const Eigen::Quaternionf &state_q,
                             double sigma_pos, double sigma_rot)
  {
    if (!initialized_)
      return false;

    Eigen::Matrix<float, 6, 15> H = Eigen::Matrix<float, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
    H.block<3, 3>(3, 6) = Eigen::Matrix3f::Identity();

    float sp2 = static_cast<float>(sigma_pos * sigma_pos);
    float sr2 = static_cast<float>(sigma_rot * sigma_rot);
    Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Zero();
    R.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * sp2;
    R.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * sr2;

    Eigen::Matrix<float, 6, 1> y;
    y.segment<3>(0) = z_pos - state_p;

    Eigen::Quaternionf qe = state_q.conjugate() * z_quat;
    float sgn = (qe.w() < 0) ? -1.0f : 1.0f;
    y.segment<3>(3) = 2.0f * sgn * qe.vec();

    Eigen::Matrix<float, 6, 6> S = H * P_ * H.transpose() + R;
    Eigen::Matrix<float, 6, 6> S_inv = S.inverse();

    float mahal2 = (y.transpose() * S_inv * y)(0, 0);
    last_mahal_ = mahal2;

    if (params_.gate_threshold > 0 && mahal2 > params_.gate_threshold)
    {
      dx_.setZero();
      consecutive_rejects_++;
      return false;
    }

    consecutive_rejects_ = 0;

    Eigen::Matrix<float, 15, 6> K = P_ * H.transpose() * S_inv;
    dx_ = K * y;

    Eigen::Matrix<float, 15, 15> I_KH = Eigen::Matrix<float, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    P_ = 0.5f * (P_ + P_.transpose());

    return true;
  }

  std::pair<double, double> ErrorStateEkf::computeMeasurementNoise(
      const EkfParams &params,
      double fitness_score,
      double spaciousness,
      double scatter)
  {
    double capped_fitness = std::isfinite(fitness_score)
                                ? std::min(fitness_score, params.max_fitness)
                                : params.max_fitness;

    double fitness_scale = 1.0 + params.fitness_weight * capped_fitness;
    double space_scale = 1.0 + params.space_weight * (spaciousness / 5.0);

    // Degeneracy: low scatter = degenerate geometry (tunnel) → inflate sigma
    double clamped_scatter = std::clamp(scatter, 0.0, 1.0);
    double degeneracy_scale = 1.0 + params.degeneracy_weight * (1.0 - clamped_scatter);

    double sigma_pos = params.sigma_pos_base * fitness_scale * space_scale * degeneracy_scale;
    double sigma_rot = params.sigma_rot_base * fitness_scale * degeneracy_scale;

    sigma_pos = std::clamp(sigma_pos, params.sigma_pos_min, params.sigma_pos_max);
    sigma_rot = std::clamp(sigma_rot, params.sigma_rot_min, params.sigma_rot_max);

    return {sigma_pos, sigma_rot};
  }

  double ErrorStateEkf::computeScanScatter(
      const pcl::PointCloud<PointType>::ConstPtr &scan)
  {
    if (!scan || scan->size() < 50)
      return 0.5;

    // Single-pass: compute mean and covariance of scan point positions
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    int n = 0;
    for (const auto &pt : scan->points)
    {
      if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      {
        mean += Eigen::Vector3d(pt.x, pt.y, pt.z);
        n++;
      }
    }
    if (n < 50)
      return 0.5;
    mean /= n;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &pt : scan->points)
    {
      if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      {
        Eigen::Vector3d diff(pt.x - mean.x(), pt.y - mean.y(), pt.z - mean.z());
        cov += diff * diff.transpose();
      }
    }
    cov /= (n - 1);

    // Eigenvalues → scatter = λ_min / λ_max
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success)
      return 0.5;

    Eigen::Vector3d evals = solver.eigenvalues();
    double lambda_max = std::max(evals(2), 1e-10);
    double lambda_min = std::max(evals(0), 0.0);

    return lambda_min / lambda_max;
  }

} // namespace c_lio
