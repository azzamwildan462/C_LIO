#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace c_lio
{

  struct ClassicModeKFParams
  {
    // Complementary filter blend weight applied each update() call — 0 =
    // never trust the registration measurement (pure dead-reckoning), 1 =
    // snap fully to the measurement every time. No covariance bookkeeping.
    double comp_alpha_pos = 0.05;
    double comp_alpha_rot = 0.05;
  };

  // Standalone position+orientation complementary filter backing
  // "cls_odom_filtered" for c_lio's classic localization mode. Deliberately
  // NOT a covariance-based Kalman filter (an earlier draft was, but a fixed-
  // weight complementary blend is simpler and easier to reason about) — no
  // velocity/bias state either. predict() dead-reckons forward from a
  // caller-supplied delta (from update_unlocalized_odom()'s differencing);
  // update() blends toward a frozen-map registration pose, gated externally
  // by apply_corr_gate() before being called — this class has no gating
  // logic of its own.
  class ClassicModeKF
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ClassicModeKF() = default;

    void init(const ClassicModeKFParams &params,
              const Eigen::Vector3f &p0, const Eigen::Quaternionf &q0);

    // Predict step ("cls_odom_filtered += delta_unlocalized_odom"). Composes
    // delta_p/delta_q (expressed in the filter's own previous-pose frame)
    // onto the current fused pose.
    void predict(const Eigen::Vector3f &delta_p, const Eigen::Quaternionf &delta_q);

    // Update step ("fuse_odom()") — complementary-blend toward a
    // registration-measured global pose. Caller (apply_corr_gate()) must
    // gate this externally; this class always applies the blend
    // unconditionally when called.
    void update(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q);

    const Eigen::Vector3f &position() const { return p_; }
    const Eigen::Quaternionf &orientation() const { return q_; }
    bool isInitialized() const { return initialized_; }

  private:
    ClassicModeKFParams params_;
    Eigen::Vector3f p_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf q_ = Eigen::Quaternionf::Identity();
    bool initialized_ = false;
  };

} // namespace c_lio
