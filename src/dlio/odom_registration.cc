#include "dlio/odom.h"
#include "dlio/utils.h"

void dlio::OdomNode::initializeInputTarget()
{

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  if (this->use_gicp_)
  {
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  }
  else
  {
    this->keyframe_normals.push_back(nullptr);
  }
  this->keyframe_transformations.push_back(this->T_corr);
}

void dlio::OdomNode::setInputSource()
{
  if (this->registration_method_ == "robust_icp")
  {
    this->robust_icp_.setInputSource(this->current_scan);
  }
  else if (this->use_gicp_)
  {
    this->gicp.setInputSource(this->current_scan);
    this->gicp.calculateSourceCovariances();
  }
  else
  {
    this->ndt.setInputSource(this->current_scan);
  }
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
                "[DEEP] getNextPose: submap_ready=%d, submap_changed=%d, use_gicp=%d",
                (int)this->new_submap_is_ready, (int)this->submap_hasChanged.load(), (int)this->use_gicp_);

  if (this->new_submap_is_ready && this->submap_hasChanged)
  {
    if (this->registration_method_ == "robust_icp")
    {
      this->robust_icp_.setInputTarget(this->submap_cloud);
      this->robust_icp_.setTargetKdTree(this->submap_kdtree);
    }
    else if (this->use_gicp_)
    {
      if (this->deep_debug_)
        RCLCPP_INFO(this->get_logger(),
                    "[DEEP] getNextPose: registerInputTarget, submap_cloud=%zu, submap_normals=%zu, kdtree=%p",
                    this->submap_cloud ? this->submap_cloud->points.size() : 0,
                    this->submap_normals ? this->submap_normals->size() : 0,
                    (void *)this->submap_kdtree.get());

      // Set the current global submap as the target cloud
      this->gicp.registerInputTarget(this->submap_cloud);

      // Set submap kdtree
      this->gicp.target_kdtree_ = this->submap_kdtree;

      // Set target cloud's normals as submap normals
      this->gicp.setTargetCovariances(this->submap_normals);

      if (this->deep_debug_)
        RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: target registered OK");
    }
    else
    {
      this->ndt.setInputTarget(this->submap_cloud);
    }

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();

  if (this->registration_method_ == "robust_icp")
  {
    this->robust_icp_.align(*aligned);
    this->T_corr = this->robust_icp_.getFinalTransformation();
    this->last_fitness_ = this->robust_icp_.getFitnessScore(1.0);
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: robust_icp done, fitness=%.4f", this->last_fitness_);
  }
  else if (this->use_gicp_)
  {
    this->gicp.align(*aligned);
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: gicp.align done");
    this->T_corr = this->gicp.getFinalTransformation();
    // Manual fitness: PCL's getFitnessScore() crashes because tree_ is null
    {
      double score = 0.0;
      int nr = 0;
      pcl::PointCloud<PointType> transformed;
      pcl::transformPointCloud(*this->gicp.getInputSource(), transformed, this->T_corr);
      std::vector<int> nn_idx(1);
      std::vector<float> nn_dist(1);
      for (const auto &pt : transformed.points)
      {
        if (!std::isfinite(pt.x))
          continue;
        this->gicp.target_kdtree_->nearestKSearch(pt, 1, nn_idx, nn_dist);
        if (nn_dist[0] <= 1.0)
        {
          score += nn_dist[0];
          nr++;
        }
      }
      this->last_fitness_ = (nr > 0) ? (score / nr) : std::numeric_limits<double>::max();
    }
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: gicp fitness=%.4f", this->last_fitness_);
  }
  else
  {
    this->ndt.align(*aligned);
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: ndt.align done");
    this->T_corr = this->ndt.getFinalTransformation();
    try
    {
      this->last_fitness_ = this->ndt.getFitnessScore(1.0);
    }
    catch (...)
    {
      this->last_fitness_ = this->ndt.hasConverged() ? 0.1 : 1.0;
      RCLCPP_WARN(this->get_logger(), "[odom] NDT getFitnessScore failed, using fallback");
    }
  }

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: fitness=%.4f", this->last_fitness_);

  this->T_prev_ = this->T; // store for constant velocity model (before update)
  this->T = this->T_corr * this->T_prior;

  if (this->deep_debug_)
  {
    RCLCPP_INFO(this->get_logger(),
                "[DEEP] Z-debug: T_prior_z=%.4f T_corr_z=%.4f T_final_z=%.4f fitness=%.4f src=%zu tgt=%zu",
                this->T_prior(2, 3), this->T_corr(2, 3), this->T(2, 3), this->last_fitness_,
                this->gicp.getInputSource() ? this->gicp.getInputSource()->size() : 0,
                this->gicp.getInputTarget() ? this->gicp.getInputTarget()->size() : 0);
  }

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: propagateGICP done");

  // Geometric observer update
  this->updateState();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] getNextPose: updateState done");
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

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp)
    {
      // Time between previous IMU sample and given timestamp
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

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v * dt + 0.5 * a0 * dt * dt + (1 / 6.) * j_dt * dt * dt;

    // Velocity
    v += a0 * dt + 0.5 * j_dt * dt;

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

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0] * dt + 0.5 * dt * dt * world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1] * dt + 0.5 * dt * dt * world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2] * dt + 0.5 * dt * dt * (world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0] * dt;
  this->state.v.lin.w[1] += world_accel[1] * dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_) * dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.x() += 0.5 * dt * tmp.x();
  this->state.q.y() += 0.5 * dt * tmp.y();
  this->state.q.z() += 0.5 * dt * tmp.z();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void dlio::OdomNode::updateState()
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

  if (this->registration_method_ == "robust_icp")
  {
    this->robust_icp_.setMaxCorrespondenceDistance(den);
  }
  else if (this->use_gicp_)
  {
    this->gicp.setMaxCorrespondenceDistance(den);
  }

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
}
