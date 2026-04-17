#include "dlio/odom/odom.h"
#include "dlio/odom/utils.h"

void dlio::OdomNode::initializeInputTarget()
{

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->engine_.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);
}

void dlio::OdomNode::setInputSource()
{
  this->engine_.setInputSource(this->current_scan);
}

void dlio::OdomNode::getNextPose()
{

  // Check if the new submap is ready to be used
  if (this->submap_future.valid())
  {
    this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
  }

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(),
                "[DEEP] getNextPose: submap_ready=%d, submap_changed=%d, method=%s",
                (int)this->new_submap_is_ready, (int)this->submap_hasChanged.load(),
                this->registration_method_.c_str());

  if (this->new_submap_is_ready && this->submap_hasChanged)
  {
    this->engine_.registerInputTarget(this->submap_cloud);
    if (this->engine_.needsKdTree())
      this->engine_.setTargetKdTree(this->submap_kdtree);
    if (this->engine_.needsCovariances())
      this->engine_.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;

    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: target registered OK");
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->engine_.align(*aligned);
  this->T_corr = this->engine_.getFinalTransformation();
  this->last_fitness_ = this->engine_.getFitnessScore(1.0);
  this->gicp_hasConverged = this->engine_.hasConverged();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: align done, fitness=%.4f", this->last_fitness_);

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: fitness=%.4f", this->last_fitness_);

  this->T_prev_ = this->T; // store for constant velocity model (before update)
  this->T = this->T_corr * this->T_prior;

  if (this->deep_debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[DEEP] Z-debug: T_prior_z=%.4f T_corr_z=%.4f T_final_z=%.4f fitness=%.4f src=%zu tgt=%zu",
                this->T_prior(2, 3), this->T_corr(2, 3), this->T(2, 3), this->last_fitness_,
                this->current_scan ? this->current_scan->size() : 0,
                this->submap_cloud ? this->submap_cloud->size() : 0);
  }

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: propagateGICP done");

  // NOTE: updateState() is now called from callbackPointCloud() after gate evaluation
}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator &begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator &end_imu_it)
{

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time)
  {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]
                            { return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time)
  {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time)
  {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end())
  {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double> &sorted_timestamps)
{

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front())
  {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false)
  {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas &f1 = *begin_imu_it;
  const ImuMeas &f2 = *(begin_imu_it + 1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5 * alpha * idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf(
      q_init.w() - 0.5 * (q_init.x() * omega_i[0] + q_init.y() * omega_i[1] + q_init.z() * omega_i[2]) * idt,
      q_init.x() + 0.5 * (q_init.w() * omega_i[0] - q_init.z() * omega_i[1] + q_init.y() * omega_i[2]) * idt,
      q_init.y() + 0.5 * (q_init.z() * omega_i[0] + q_init.w() * omega_i[1] - q_init.x() * omega_i[2]) * idt,
      q_init.z() + 0.5 * (q_init.x() * omega_i[1] - q_init.y() * omega_i[0] + q_init.w() * omega_i[2]) * idt);
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5 * alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2(
      q_init.w() - 0.5 * (q_init.x() * omega[0] + q_init.y() * omega[1] + q_init.z() * omega[2]) * dt,
      q_init.x() + 0.5 * (q_init.w() * omega[0] - q_init.z() * omega[1] + q_init.y() * omega[2]) * dt,
      q_init.y() + 0.5 * (q_init.z() * omega[0] + q_init.w() * omega[1] - q_init.x() * omega[2]) * dt,
      q_init.z() + 0.5 * (q_init.x() * omega[1] - q_init.y() * omega[0] + q_init.w() * omega[2]) * dt);
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1 * idt + 0.5 * j * idt * idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init * idt + 0.5 * a1 * idt * idt + (1 / 6.) * j * idt * idt * idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double> &sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it)
{

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++)
  {

    const ImuMeas &f0 = *prev_imu_it;
    const ImuMeas &f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5 * alpha_dt;

    // Orientation
    q = Eigen::Quaternionf(
        q.w() - 0.5 * (q.x() * omega[0] + q.y() * omega[1] + q.z() * omega[2]) * dt,
        q.x() + 0.5 * (q.w() * omega[0] - q.z() * omega[1] + q.y() * omega[2]) * dt,
        q.y() + 0.5 * (q.z() * omega[0] + q.w() * omega[1] - q.x() * omega[2]) * dt,
        q.z() + 0.5 * (q.x() * omega[1] - q.y() * omega[0] + q.w() * omega[2]) * dt);
    q.normalize();

    // Acceleration & velocity propagation
    Eigen::Vector3f a0 = a;

    if (this->ext_odom_enabled_ && this->ext_odom_received_.load())
    {
      // External odom: use odom velocity, skip accel integration
      Eigen::Vector3f vel_body;
      {
        std::lock_guard<std::mutex> odom_lock(this->ext_odom_mtx_);
        vel_body = this->ext_odom_vel_body_;
      }
      v = q * vel_body; // body → world
      a = Eigen::Vector3f::Zero();
    }
    else
    {
      a = q._transformVector(f.lin_accel);
      a[2] -= this->gravity_;
    }

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp)
    {
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5 * alpha * idt;

      // Orientation
      Eigen::Quaternionf q_i(
          q.w() - 0.5 * (q.x() * omega_i[0] + q.y() * omega_i[1] + q.z() * omega_i[2]) * idt,
          q.x() + 0.5 * (q.w() * omega_i[0] - q.z() * omega_i[1] + q.y() * omega_i[2]) * idt,
          q.y() + 0.5 * (q.z() * omega_i[0] + q.w() * omega_i[1] - q.x() * omega_i[2]) * idt,
          q.z() + 0.5 * (q.x() * omega_i[1] - q.y() * omega_i[0] + q.w() * omega_i[2]) * idt);
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v * idt + 0.5 * a0 * idt * idt + (1 / 6.) * j * idt * idt * idt;

      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;
      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v * dt + 0.5 * a0 * dt * dt + (1 / 6.) * j_dt * dt * dt;

    // Velocity
    if (!(this->ext_odom_enabled_ && this->ext_odom_received_.load()))
    {
      v += a0 * dt + 0.5 * j_dt * dt;
    }

    // Motion model constraint on velocity
    this->applyMotionModelConstraintImu(v, q);

    prev_imu_it = imu_it;
  }

  return imu_se3;
}

void dlio::OdomNode::propagateGICP()
{

  this->lidarPose.p << this->T(0, 3), this->T(1, 3), this->T(2, 3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0, 0), this->T(0, 1), this->T(0, 2),
      this->T(1, 0), this->T(1, 1), this->T(1, 2),
      this->T(2, 0), this->T(2, 1), this->T(2, 2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  q.w() /= norm;
  q.x() /= norm;
  q.y() /= norm;
  q.z() /= norm;
  this->lidarPose.q = q;
}

void dlio::OdomNode::propagateState()
{

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  double dt = this->imu_meas.dt;

  // Sanity check on dt
  if (dt <= 0 || dt > 1.0)
  {
    return;
  }

  if (this->imu_preintegration_mode_ == "gtsam" && this->imu_preintegration_)
  {
    // --- GTSAM preintegration mode ---
    gtsam::Vector3 accel, gyro;

    if (this->ext_odom_enabled_ && this->ext_odom_received_.load())
    {
      Eigen::Vector3f vel_body;
      {
        std::lock_guard<std::mutex> odom_lock(this->ext_odom_mtx_);
        vel_body = this->ext_odom_vel_body_;
      }
      Eigen::Vector3f vel_world = this->state.q * vel_body;
      this->state.v.lin.w = vel_world;
      this->state.v.lin.b = vel_body;

      accel = this->imu_meas.lin_accel.cast<double>();
      gyro = this->imu_meas.ang_vel.cast<double>();
    }
    else
    {
      accel = this->imu_meas.lin_accel.cast<double>();
      gyro = this->imu_meas.ang_vel.cast<double>();
    }

    this->imu_preintegration_->integrateMeasurement(accel, gyro, dt);

    if (this->gtsam_imu_initialized_)
    {
      gtsam::NavState predicted = this->imu_preintegration_->predict(
          this->gtsam_nav_state_, this->gtsam_bias_);

      gtsam::Quaternion gq = predicted.quaternion();
      this->state.q = Eigen::Quaternionf(gq.w(), gq.x(), gq.y(), gq.z());
      this->state.q.normalize();

      if (this->ext_odom_enabled_ && this->ext_odom_received_.load())
      {
        Eigen::Vector3f vel_world = this->state.q * this->state.v.lin.b;
        this->state.p += vel_world * static_cast<float>(dt);
      }
      else
      {
        this->state.p = predicted.position().cast<float>();
        this->state.v.lin.w = predicted.velocity().cast<float>();
        this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;
      }
    }
  }
  else
  {
    // --- Simple dead reckoning mode ---
    Eigen::Quaternionf qhat = this->state.q;

    if (this->ext_odom_enabled_ && this->ext_odom_received_.load())
    {
      Eigen::Vector3f vel_body;
      {
        std::lock_guard<std::mutex> odom_lock(this->ext_odom_mtx_);
        vel_body = this->ext_odom_vel_body_;
      }
      Eigen::Vector3f vel_world = qhat * vel_body;
      this->state.p += vel_world * static_cast<float>(dt);
      this->state.v.lin.w = vel_world;
      this->state.v.lin.b = vel_body;
    }
    else
    {
      Eigen::Vector3f world_accel = qhat._transformVector(this->imu_meas.lin_accel);
      float dtf = static_cast<float>(dt);

      this->state.p[0] += this->state.v.lin.w[0] * dtf + 0.5f * dtf * dtf * world_accel[0];
      this->state.p[1] += this->state.v.lin.w[1] * dtf + 0.5f * dtf * dtf * world_accel[1];
      this->state.p[2] += this->state.v.lin.w[2] * dtf + 0.5f * dtf * dtf * (world_accel[2] - this->gravity_);

      this->state.v.lin.w[0] += world_accel[0] * dtf;
      this->state.v.lin.w[1] += world_accel[1] * dtf;
      this->state.v.lin.w[2] += (world_accel[2] - this->gravity_) * dtf;
      this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

      this->imu_only_vel_w_[0] += world_accel[0] * dtf;
      this->imu_only_vel_w_[1] += world_accel[1] * dtf;
      this->imu_only_vel_w_[2] += (world_accel[2] - this->gravity_) * dtf;
      this->imu_only_vel_b_ = this->state.q.toRotationMatrix().inverse() * this->imu_only_vel_w_;
    }

    // Gyro quaternion integration
    Eigen::Quaternionf omega;
    omega.w() = 0;
    omega.vec() = this->imu_meas.ang_vel;
    Eigen::Quaternionf tmp = qhat * omega;
    float dtf = static_cast<float>(dt);
    this->state.q.w() += 0.5f * dtf * tmp.w();
    this->state.q.x() += 0.5f * dtf * tmp.x();
    this->state.q.y() += 0.5f * dtf * tmp.y();
    this->state.q.z() += 0.5f * dtf * tmp.z();
    this->state.q.normalize();
  }

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

  // Motion model constraint (clamp velocity/angular rate to physical limits)
  this->applyMotionModelConstraint();

  // Covariance propagation for KF/EKF
  if (this->fusion_method_ == dlio::FusionMethod::KF)
  {
    this->propagateStateKF(dt);
  }
  else if (this->fusion_method_ == dlio::FusionMethod::EKF)
  {
    this->ekf_.predict(dt, this->imu_meas.lin_accel, this->imu_meas.ang_vel,
                       this->state.q, static_cast<float>(this->gravity_));
  }
}

void dlio::OdomNode::updateState()
{
  switch (this->fusion_method_)
  {
  case dlio::FusionMethod::KF:
    this->updateStateKF();
    break;
  case dlio::FusionMethod::EKF:
    this->updateStateEKF();
    break;
  case dlio::FusionMethod::GEO:
  default:
    this->updateStateGeo();
    break;
  }
}

void dlio::OdomNode::updateStateGeo()
{

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate() * qin;

  double sgn = 1.;
  if (qe.w() < 0)
  {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn * qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Apply geometric observer corrections directly to state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[odom] updateState: err=[%.4f,%.4f,%.4f] |err|=%.4f | bias_a=[%.4f,%.4f,%.4f] bias_g=[%.4f,%.4f,%.4f]",
                err[0], err[1], err[2], err.norm(),
                this->state.b.accel[0], this->state.b.accel[1], this->state.b.accel[2],
                this->state.b.gyro[0], this->state.b.gyro[1], this->state.b.gyro[2]);
  }
}

void dlio::OdomNode::updateStateKF()
{
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;

  // Innovation vector (6x1): [pos_error(3), rot_error(3)]
  Eigen::Matrix<float, 6, 1> y;
  y.segment<3>(0) = pin - this->state.p;

  Eigen::Quaternionf qe = this->state.q.conjugate() * qin;
  float sgn = (qe.w() < 0) ? -1.0f : 1.0f;
  y.segment<3>(3) = 2.0f * sgn * qe.vec();

  // Observation matrix H (6x15): identity blocks for position and rotation error
  Eigen::Matrix<float, 6, 15> H = Eigen::Matrix<float, 6, 15>::Zero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity(); // position
  H.block<3, 3>(3, 6) = Eigen::Matrix3f::Identity(); // rotation error

  // Measurement noise R (6x6)
  float sp2 = static_cast<float>(this->kf_sigma_pos_meas_ * this->kf_sigma_pos_meas_);
  float sr2 = static_cast<float>(this->kf_sigma_rot_meas_ * this->kf_sigma_rot_meas_);
  Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Zero();
  R.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * sp2;
  R.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * sr2;

  // Innovation covariance S = H*P*H^T + R
  Eigen::Matrix<float, 6, 6> S = H * this->kf_P_ * H.transpose() + R;

  // Kalman gain K = P * H^T * S^{-1}
  Eigen::Matrix<float, 6, 6> S_inv = S.inverse();
  Eigen::Matrix<float, 15, 6> K = this->kf_P_ * H.transpose() * S_inv;

  // State correction
  Eigen::Matrix<float, 15, 1> dx = K * y;

  // Apply correction to nominal state
  this->state.p += dx.segment<3>(0);
  this->state.v.lin.w += dx.segment<3>(3);

  // Quaternion: apply small rotation dtheta
  Eigen::Vector3f dtheta = dx.segment<3>(6);
  Eigen::Quaternionf dq(1.0f, dtheta.x() / 2.0f, dtheta.y() / 2.0f, dtheta.z() / 2.0f);
  dq.normalize();
  this->state.q = (this->state.q * dq).normalized();

  // Bias corrections
  this->state.b.accel += dx.segment<3>(9);
  this->state.b.gyro += dx.segment<3>(12);
  this->state.b.accel = this->state.b.accel.array().min(this->geo_abias_max_).max(-this->geo_abias_max_);
  this->state.b.gyro = this->state.b.gyro.array().min(this->geo_gbias_max_).max(-this->geo_gbias_max_);

  // Joseph-form covariance update: P = (I-KH)*P*(I-KH)^T + K*R*K^T
  Eigen::Matrix<float, 15, 15> I_KH = Eigen::Matrix<float, 15, 15>::Identity() - K * H;
  this->kf_P_ = I_KH * this->kf_P_ * I_KH.transpose() + K * R * K.transpose();
  this->kf_P_ = 0.5f * (this->kf_P_ + this->kf_P_.transpose());

  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[odom] updateStateKF: innovation=[%.4f,%.4f,%.4f] |y_pos|=%.4f |y_rot|=%.4f",
                y(0), y(1), y(2), y.segment<3>(0).norm(), y.segment<3>(3).norm());
  }
}

void dlio::OdomNode::updateStateEKF()
{
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  // Compute adaptive measurement noise from fitness + spaciousness + scatter
  float sp;
  {
    std::lock_guard<std::mutex> mlock(this->metrics_mtx_);
    sp = this->metrics.spaciousness.empty() ? 5.0f : this->metrics.spaciousness.back();
  }
  double scatter = dlio::ErrorStateEkf::computeScanScatter(this->current_scan);

  auto [sigma_pos, sigma_rot] = dlio::ErrorStateEkf::computeMeasurementNoise(
      this->ekf_params_, this->last_fitness_, static_cast<double>(sp), scatter);

  // EKF update (includes Mahalanobis gate)
  bool accepted = this->ekf_.update(this->lidarPose.p, this->lidarPose.q,
                                    this->state.p, this->state.q,
                                    sigma_pos, sigma_rot);

  if (accepted)
  {
    const auto &dx = this->ekf_.getDeltaState();
    this->state.p += dx.segment<3>(0);
    this->state.v.lin.w += dx.segment<3>(3);

    Eigen::Vector3f dtheta = dx.segment<3>(6);
    Eigen::Quaternionf dq(1.0f, dtheta.x() / 2.0f, dtheta.y() / 2.0f, dtheta.z() / 2.0f);
    dq.normalize();
    this->state.q = (this->state.q * dq).normalized();

    this->state.b.accel += dx.segment<3>(9);
    this->state.b.gyro += dx.segment<3>(12);
    this->state.b.accel = this->state.b.accel.array()
                              .min(this->ekf_params_.abias_max)
                              .max(-this->ekf_params_.abias_max);
    this->state.b.gyro = this->state.b.gyro.array()
                             .min(this->ekf_params_.gbias_max)
                             .max(-this->ekf_params_.gbias_max);
  }

  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

  if (this->debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[odom] updateStateEKF: accepted=%d mahal=%.2f sigma_p=%.4f sigma_r=%.4f rejects=%d",
                (int)accepted, this->ekf_.lastMahalanobis(),
                sigma_pos, sigma_rot, this->ekf_.consecutiveRejects());
  }
}

void dlio::OdomNode::propagateStateKF(double dt)
{
  // Linear covariance propagation (called inside propagateState which holds geo.mtx)
  float dtf = static_cast<float>(dt);
  Eigen::Matrix3f R = this->state.q.toRotationMatrix();

  // State transition matrix F (15x15)
  Eigen::Matrix<float, 15, 15> F = Eigen::Matrix<float, 15, 15>::Identity();
  F.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dtf; // dp/dv = I*dt

  // Simplified linear model: skip rotation-dependent terms for KF
  // (EKF handles those properly; KF is a simpler alternative)
  F.block<3, 3>(3, 9) = -R * dtf;                            // dv/dba = -R*dt
  F.block<3, 3>(6, 12) = -Eigen::Matrix3f::Identity() * dtf; // dtheta/dbg = -I*dt

  // Process noise input matrix G (15x12)
  Eigen::Matrix<float, 15, 12> G = Eigen::Matrix<float, 15, 12>::Zero();
  G.block<3, 3>(3, 0) = -R;
  G.block<3, 3>(6, 3) = -Eigen::Matrix3f::Identity();
  G.block<3, 3>(9, 6) = Eigen::Matrix3f::Identity();
  G.block<3, 3>(12, 9) = Eigen::Matrix3f::Identity();

  // Continuous-time spectral densities
  float sa2 = this->kf_sigma_accel_ * this->kf_sigma_accel_;
  float sg2 = this->kf_sigma_gyro_ * this->kf_sigma_gyro_;
  float sba2 = this->kf_sigma_accel_bias_ * this->kf_sigma_accel_bias_;
  float sbg2 = this->kf_sigma_gyro_bias_ * this->kf_sigma_gyro_bias_;

  Eigen::Matrix<float, 12, 12> Qi = Eigen::Matrix<float, 12, 12>::Zero();
  Qi.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * sa2;
  Qi.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * sg2;
  Qi.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * sba2;
  Qi.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * sbg2;

  this->kf_P_ = F * this->kf_P_ * F.transpose() + G * Qi * G.transpose() * dtf;

  // Periodic symmetrization for numerical stability
  static int kf_prop_count = 0;
  if (++kf_prop_count % 100 == 0)
  {
    this->kf_P_ = 0.5f * (this->kf_P_ + this->kf_P_.transpose());

    // Clamp diagonal to prevent covariance explosion
    // Max std: pos=100m, vel=50m/s, rot=1rad, abias=5m/s², gbias=0.5rad/s
    Eigen::Matrix<float, 15, 1> max_var;
    max_var << 1e4, 1e4, 1e4, // position variance (100m)^2
        2500, 2500, 2500,     // velocity variance (50m/s)^2
        1, 1, 1,              // rotation variance (1rad)^2
        25, 25, 25,           // accel bias variance (5m/s²)^2
        0.25, 0.25, 0.25;     // gyro bias variance (0.5rad/s)^2
    for (int i = 0; i < 15; i++)
    {
      if (this->kf_P_(i, i) > max_var(i))
        this->kf_P_(i, i) = max_var(i);
    }
  }
}

bool dlio::OdomNode::evaluatePoseGate()
{
  if (!this->gate_enabled_)
    return true; // backward compatible: no gating

  // Gate 0: GICP convergence check
  if (!this->gicp_hasConverged.load())
  {
    return false;
  }

  // Gate 0b: T_corr sanity — if vehicle is moving but T_corr ≈ identity, GICP failed silently
  {
    float t_corr_trans = Eigen::Vector3f(this->T_corr(0, 3), this->T_corr(1, 3), this->T_corr(2, 3)).norm();
    float speed = this->state.v.lin.b.norm();
    float dt = this->scan_stamp - this->prev_scan_stamp;
    float expected_displacement = speed * dt;

    // If we're moving (>1m/s) and expected displacement is significant (>0.1m),
    // but T_corr shows near-zero correction, GICP likely failed silently
    if (speed > 1.0f && expected_displacement > 0.1f && t_corr_trans < 0.01f)
    {
      return false;
    }
  }

  // Gate 1: Fitness score (skip if infinite/nan — GICP failed completely)
  if (std::isfinite(this->last_fitness_) && this->last_fitness_ > this->gate_fitness_threshold_)
  {
    return false;
  }
  if (!std::isfinite(this->last_fitness_))
  {
    return false;
  }

  // Gate 2: Structure complexity (spaciousness)
  float sp;
  {
    std::lock_guard<std::mutex> lock(this->metrics_mtx_);
    sp = this->metrics.spaciousness.empty() ? 10.0f : this->metrics.spaciousness.back();
  }
  if (sp < this->gate_min_spaciousness_)
  {
    return false;
  }

  // Gate 3: Max translation jump (vs current state pose)
  float trans = (this->lidarPose.p - this->state.p).norm();
  if (trans > this->gate_max_translation_)
  {
    return false;
  }

  // Gate 4: Max rotation jump (vs current state orientation)
  Eigen::Quaternionf dq = this->state.q.conjugate() * this->lidarPose.q;
  double angle_deg = 2.0 * std::acos(std::min(1.0f, std::abs(dq.w()))) * 180.0 / M_PI;
  if (angle_deg > this->gate_max_rotation_deg_)
  {
    return false;
  }

  return true;
}

void dlio::OdomNode::applyMotionModelConstraint()
{
  // Called from propagateState() which already holds geo.mtx
  if (this->motion_model_type_ == dlio::MotionModelType::NONE)
    return;

  // Convert world-frame velocity to body-frame
  Eigen::Vector3f v_body = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;
  Eigen::Vector3f omega = this->state.v.ang.b;

  const auto &p = this->mm_params_;

  switch (this->motion_model_type_)
  {
  case dlio::MotionModelType::ACKERMANN:
    v_body[0] = std::clamp(v_body[0], -p.ack_max_rev_vel, p.ack_max_fwd_vel);
    v_body[1] = std::clamp(v_body[1], -p.ack_max_lat_vel, p.ack_max_lat_vel);
    v_body[2] = std::clamp(v_body[2], -p.ack_max_vert_vel, p.ack_max_vert_vel);
    omega[0] = std::clamp(omega[0], -p.ack_max_roll_rate, p.ack_max_roll_rate);
    omega[1] = std::clamp(omega[1], -p.ack_max_pitch_rate, p.ack_max_pitch_rate);
    omega[2] = std::clamp(omega[2], -p.ack_max_yaw_rate, p.ack_max_yaw_rate);
    break;

  case dlio::MotionModelType::DIFF_DRIVE:
    v_body[0] = std::clamp(v_body[0], -p.dd_max_rev_vel, p.dd_max_fwd_vel);
    v_body[1] = std::clamp(v_body[1], -p.dd_max_lat_vel, p.dd_max_lat_vel);
    v_body[2] = std::clamp(v_body[2], -p.dd_max_vert_vel, p.dd_max_vert_vel);
    omega[0] = std::clamp(omega[0], -p.dd_max_roll_rate, p.dd_max_roll_rate);
    omega[1] = std::clamp(omega[1], -p.dd_max_pitch_rate, p.dd_max_pitch_rate);
    omega[2] = std::clamp(omega[2], -p.dd_max_yaw_rate, p.dd_max_yaw_rate);
    break;

  case dlio::MotionModelType::HOLONOMIC:
    v_body[0] = std::clamp(v_body[0], -p.holo_max_horiz_vel, p.holo_max_horiz_vel);
    v_body[1] = std::clamp(v_body[1], -p.holo_max_horiz_vel, p.holo_max_horiz_vel);
    v_body[2] = std::clamp(v_body[2], -p.holo_max_vert_vel, p.holo_max_vert_vel);
    omega[0] = std::clamp(omega[0], -p.holo_max_roll_rate, p.holo_max_roll_rate);
    omega[1] = std::clamp(omega[1], -p.holo_max_pitch_rate, p.holo_max_pitch_rate);
    omega[2] = std::clamp(omega[2], -p.holo_max_yaw_rate, p.holo_max_yaw_rate);
    break;

  default:
    return;
  }

  // Write back to state
  this->state.v.lin.b = v_body;
  this->state.v.lin.w = this->state.q.toRotationMatrix() * v_body;
  this->state.v.ang.b = omega;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * omega;
}

void dlio::OdomNode::applyMotionModelConstraintImu(Eigen::Vector3f &v, const Eigen::Quaternionf &q)
{
  // Lightweight version for integrateImuInternal() — only clamps velocity
  if (this->motion_model_type_ == dlio::MotionModelType::NONE)
    return;

  Eigen::Vector3f v_body = q.toRotationMatrix().inverse() * v;
  const auto &p = this->mm_params_;

  switch (this->motion_model_type_)
  {
  case dlio::MotionModelType::ACKERMANN:
    v_body[0] = std::clamp(v_body[0], -p.ack_max_rev_vel, p.ack_max_fwd_vel);
    v_body[1] = std::clamp(v_body[1], -p.ack_max_lat_vel, p.ack_max_lat_vel);
    v_body[2] = std::clamp(v_body[2], -p.ack_max_vert_vel, p.ack_max_vert_vel);
    break;
  case dlio::MotionModelType::DIFF_DRIVE:
    v_body[0] = std::clamp(v_body[0], -p.dd_max_rev_vel, p.dd_max_fwd_vel);
    v_body[1] = std::clamp(v_body[1], -p.dd_max_lat_vel, p.dd_max_lat_vel);
    v_body[2] = std::clamp(v_body[2], -p.dd_max_vert_vel, p.dd_max_vert_vel);
    break;
  case dlio::MotionModelType::HOLONOMIC:
    v_body[0] = std::clamp(v_body[0], -p.holo_max_horiz_vel, p.holo_max_horiz_vel);
    v_body[1] = std::clamp(v_body[1], -p.holo_max_horiz_vel, p.holo_max_horiz_vel);
    v_body[2] = std::clamp(v_body[2], -p.holo_max_vert_vel, p.holo_max_vert_vel);
    break;
  default:
    return;
  }

  v = q.toRotationMatrix() * v_body;
}

void dlio::OdomNode::setAdaptiveParams()
{

  // Spaciousness + Density (lock to prevent race with detached computeMetrics thread)
  float sp, den;
  {
    std::lock_guard<std::mutex> lock(this->metrics_mtx_);
    sp = this->metrics.spaciousness.back();
    den = this->metrics.density.back();
  }

  if (sp < 0.5)
  {
    sp = 0.5;
  }
  if (sp > 5.0)
  {
    sp = 5.0;
  }

  this->keyframe_thresh_dist_ = sp;

  if (den < 0.5 * this->gicp_max_corr_dist_)
  {
    den = 0.5 * this->gicp_max_corr_dist_;
  }
  if (den > 2.0 * this->gicp_max_corr_dist_)
  {
    den = 2.0 * this->gicp_max_corr_dist_;
  }

  if (sp < 5.0)
  {
    den = 0.5 * this->gicp_max_corr_dist_;
  };
  if (sp > 5.0)
  {
    den = 2.0 * this->gicp_max_corr_dist_;
  };

  this->engine_.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
}
