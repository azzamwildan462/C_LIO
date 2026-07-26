#include "c_lio/algorithms/classic_mode_kf.h"

void c_lio::ClassicModeKF::init(const ClassicModeKFParams &params,
                                 const Eigen::Vector3f &p0, const Eigen::Quaternionf &q0)
{
  params_ = params;
  p_ = p0;
  q_ = q0.normalized();
  initialized_ = true;
}

void c_lio::ClassicModeKF::predict(const Eigen::Vector3f &delta_p, const Eigen::Quaternionf &delta_q)
{
  if (!initialized_)
    return;

  // Compose: p_new = p_ + R(q_) * delta_p ; q_new = q_ * delta_q
  p_ += q_.toRotationMatrix() * delta_p;
  q_ = (q_ * delta_q).normalized();
}

void c_lio::ClassicModeKF::update(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q)
{
  if (!initialized_)
    return;

  // Complementary blend: nudge a fixed fraction of the way toward the
  // measurement each update, instead of a covariance-driven KF gain.
  const float a_pos = static_cast<float>(params_.comp_alpha_pos);
  const float a_rot = static_cast<float>(params_.comp_alpha_rot);

  p_ += a_pos * (meas_p - p_);
  q_ = q_.slerp(a_rot, meas_q).normalized();
}
