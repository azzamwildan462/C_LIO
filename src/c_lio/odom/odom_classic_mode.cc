/***********************************************************
 *  Classic localization sub-mode (odom/submap/method ==      *
 *  "classic"). Runs on its own timer                         *
 *  (classic_localization_timer_), fully independent of the    *
 *  scan callback — NOT a hook inside callbackPointCloud().    *
 *                                                              *
 *  "cls_odom_filtered" == classic_kf_.position()/orientation() *
 *  is the final fused output. Each tick:                       *
 *   1. update_unlocalized_odom() reads the current raw odom     *
 *      input — Case B (unlocalized_odom topic set): the topic's  *
 *      own pose. Case A (topic empty): this->T, i.e. the ORIGINAL *
 *      keyframe-based local odometry pipeline, which keeps         *
 *      running completely unmodified (map-agnostic — buildSubmap() *
 *      treats "classic" identically to "keyframe", see              *
 *      odom_keyframes.cc) purely so classic has a local motion        *
 *      estimate to differentiate.                                      *
 *   2. Differenced against the previous tick's value to get a           *
 *      delta, fed into classic_kf_.predict().                            *
 *   3. registration_to_prior_map() registers the latest scan             *
 *      (latest_scan_deskewed_body_ — motion-compensated, body frame,      *
 *      NOT the raw latest_scan_) against the frozen prior map ROI, using  *
 *      DEDICATED classic_engine_/classic_submap_* resources (kept          *
 *      separate from engine_/engine_temp_/submap_* so it never races        *
 *      the local pipeline's own async submap builder running                *
 *      concurrently in Case A).                                              *
 *   4. apply_corr_gate() gates the correction (convergence, score,            *
 *      max translation/rotation) before fuse_odom() applies it via             *
 *      classic_kf_.update().                                                    *
 *                                                                                  *
 *  "Set initial pose" (RViz "2D Pose Estimate") drains the same                    *
 *  request_fresh_reloc_ flag callbackInitialPose() sets, and snaps                  *
 *  classic_kf_ directly to the click (no ScanContext+GICP                            *
 *  relocalization — not used in classic mode; the shared fresh-reloc/                 *
 *  ScanContext-relocalization blocks in callbackPointCloud() are skipped                *
 *  entirely for classic mode, see odom_callbacks.cc).                                    *
 *                                                                                          *
 *  Output: does NOT publish anything itself. publishPose() (existing         *
 *  100Hz timer) and publishClassicToROS() (called from the scan-callback     *
 *  tail) read classic_kf_ directly when submap_method_=="classic", so       *
 *  /odom, /odom_2d, /pose, /path, TF, /deskewed all reflect classic_kf_    *
 *  — no separate topics.                                                 *
 ***********************************************************/

#include "c_lio/odom/odom.h"

void c_lio::OdomNode::callbackUnlocalizedOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(this->classic_unlocalized_odom_mtx_);
  this->classic_latest_unlocalized_odom_ = msg;
  this->classic_unlocalized_odom_seq_++;
}

std::tuple<Eigen::Vector3f, Eigen::Quaternionf, uint64_t> c_lio::OdomNode::update_unlocalized_odom()
{
  if (!this->classic_unlocalized_odom_topic_.empty())
  {
    // Case B: external odometry hardware.
    nav_msgs::msg::Odometry::SharedPtr msg;
    uint64_t seq;
    {
      std::lock_guard<std::mutex> lock(this->classic_unlocalized_odom_mtx_);
      msg = this->classic_latest_unlocalized_odom_;
      seq = this->classic_unlocalized_odom_seq_;
    }

    if (!msg)
    {
      if (this->classic_tick_count_.load() % 40 == 1)
      {
        RCLCPP_WARN(this->get_logger(),
                    "[classic] no unlocalized_odom received yet on '%s' — predict frozen this tick",
                    this->classic_unlocalized_odom_topic_.c_str());
      }
      // Same seq as last consumed => classic_localization_routine()'s
      // staleness check will correctly treat this as "nothing new".
      return {this->cls_prev_unlocalized_odom_p_, this->cls_prev_unlocalized_odom_q_,
              this->classic_last_source_seq_used_};
    }

    Eigen::Vector3f p(static_cast<float>(msg->pose.pose.position.x),
                      static_cast<float>(msg->pose.pose.position.y),
                      static_cast<float>(msg->pose.pose.position.z));
    Eigen::Quaternionf q(static_cast<float>(msg->pose.pose.orientation.w),
                         static_cast<float>(msg->pose.pose.orientation.x),
                         static_cast<float>(msg->pose.pose.orientation.y),
                         static_cast<float>(msg->pose.pose.orientation.z));
    q.normalize();
    return {p, q, seq};
  }

  // Case A: no odometry hardware wired up — reuse the ORIGINAL keyframe-based
  // local odometry pipeline's own KF/EKF-smoothed pose (state.p/q, snapshotted
  // as latest_scan_state_p_/q_ right after updateState() runs each scan — see
  // odom_callbacks.cc), NOT latest_scan_T_ (the raw pre-fusion GICP
  // correction — see latest_scan_state_p_'s comment in odom.h for why that
  // was a source of extra jitter vs "keyframe" mode's own published output).
  Eigen::Vector3f p;
  Eigen::Quaternionf q;
  uint64_t seq;
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    p = this->latest_scan_state_p_;
    q = this->latest_scan_state_q_;
    seq = this->latest_scan_seq_;
  }
  q.normalize();
  return {p, q, seq};
}

bool c_lio::OdomNode::buildClassicPriorMapRoiSubmap(const State &vehicle_state)
{
  // Classic-dedicated clone of buildPriorMapRoiSubmap() (odom_keyframes.cc) —
  // same logic, but writes into classic_submap_*/classic_roi_initialized_/
  // classic_last_roi_center_ and uses classic_engine_ directly (no separate
  // "temp" engine needed — classic's registration is lightweight/synchronous,
  // unlike the local pipeline's heavier async keyframe submap building).
  if (!this->prior_map_cloud_ || this->prior_map_cloud_->empty() || !this->prior_map_kdtree_)
  {
    this->classic_submap_hasChanged_ = false;
    return false;
  }

  if (this->classic_roi_initialized_)
  {
    double moved = (vehicle_state.p - this->classic_last_roi_center_).norm();
    if (moved < this->loc_roi_refresh_dist_)
      return true; // keep current target
  }

  PointType center;
  center.x = vehicle_state.p[0];
  center.y = vehicle_state.p[1];
  center.z = vehicle_state.p[2];

  std::vector<int> idx;
  std::vector<float> dist_sq;
  const float r2 = static_cast<float>(this->loc_roi_radius_ * this->loc_roi_radius_);
  this->prior_map_kdtree_->radiusSearch(center, r2, idx, dist_sq);

  if (idx.size() < 50)
  {
    const int k = 20000;
    int found = this->prior_map_kdtree_->nearestKSearch(
        center, std::min<int>(k, static_cast<int>(this->prior_map_cloud_->size())), idx, dist_sq);
    RCLCPP_WARN(this->get_logger(),
                "[classic] prior-map ROI sparse around (%.1f, %.1f) — using %d nearest map points instead",
                center.x, center.y, found);
    if (idx.size() < 50)
    {
      this->classic_submap_hasChanged_ = false;
      return this->classic_roi_initialized_;
    }
  }

  pcl::PointCloud<PointType>::Ptr roi = std::make_shared<pcl::PointCloud<PointType>>();
  roi->points.resize(idx.size());

  const bool need_cov = this->classic_engine_.needsCovariances();
  const bool have_precomputed =
      this->prior_map_covariances_ &&
      this->prior_map_covariances_->size() == this->prior_map_cloud_->size();

  std::shared_ptr<nano_gicp::CovarianceList> roi_covs;
  if (need_cov && have_precomputed)
    roi_covs = std::make_shared<nano_gicp::CovarianceList>(idx.size());

  for (size_t j = 0; j < idx.size(); ++j)
  {
    roi->points[j] = this->prior_map_cloud_->points[idx[j]];
    if (need_cov && have_precomputed)
      (*roi_covs)[j] = (*this->prior_map_covariances_)[idx[j]];
  }
  roi->width = roi->points.size();
  roi->height = 1;
  roi->is_dense = true;

  if (need_cov && !have_precomputed)
  {
    nano_gicp::NanoGICP<PointType, PointType> tmp_gicp;
    tmp_gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
    tmp_gicp.setInputSource(roi);
    tmp_gicp.calculateSourceCovariances();
    auto covs = tmp_gicp.getSourceCovariances();
    roi_covs = std::make_shared<nano_gicp::CovarianceList>(*covs);
  }

  this->classic_submap_cloud_ = roi;
  if (need_cov)
    this->classic_submap_normals_ = roi_covs;

  this->classic_engine_.setInputTarget(this->classic_submap_cloud_);
  if (this->classic_engine_.needsKdTree())
    this->classic_submap_kdtree_ = this->classic_engine_.getTargetKdTree();

  this->classic_last_roi_center_ = vehicle_state.p;
  this->classic_roi_initialized_ = true;
  this->classic_submap_hasChanged_ = true;
  return true;
}

bool c_lio::OdomNode::registration_to_prior_map(Eigen::Vector3f &meas_p, Eigen::Quaternionf &meas_q,
                                                double &score, bool &converged)
{
  // latest_scan_deskewed_body_ — NOT latest_scan_ (raw/undeskewed). See its
  // comment in odom.h: it's already motion-compensated and already in
  // base_link frame (baselink2lidar_T is baked in during its construction),
  // so unlike latest_scan_ it must NOT be multiplied by baselink2lidar_T
  // again below.
  pcl::PointCloud<PointType>::ConstPtr scan_body;
  {
    std::lock_guard<std::mutex> lock(this->latest_scan_mtx_);
    scan_body = this->latest_scan_deskewed_body_;
  }
  if (!scan_body || scan_body->empty())
    return false;

  Eigen::Vector3f pred_p;
  Eigen::Quaternionf pred_q;
  {
    std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
    pred_p = this->classic_kf_.position();
    pred_q = this->classic_kf_.orientation();
  }

  Eigen::Matrix4f T_predicted = Eigen::Matrix4f::Identity();
  T_predicted.block<3, 3>(0, 0) = pred_q.toRotationMatrix();
  T_predicted.block<3, 1>(0, 3) = pred_p;

  pcl::PointCloud<PointType>::Ptr scan_in_map = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(*scan_body, *scan_in_map, T_predicted);

  if (this->vf_use_)
  {
    // classic_voxel_, NOT voxel — see classic_voxel_'s comment in odom.h
    // (sharing `voxel` with preprocessPoints() is a cross-thread data race).
    this->classic_voxel_.setInputCloud(scan_in_map);
    this->classic_voxel_.filter(*scan_in_map);
  }

  State roi_state;
  roi_state.p = pred_p;
  this->buildClassicPriorMapRoiSubmap(roi_state);

  if (this->classic_submap_hasChanged_)
  {
    this->classic_engine_.registerInputTarget(this->classic_submap_cloud_);
    if (this->classic_engine_.needsKdTree() && this->classic_submap_kdtree_)
      this->classic_engine_.setTargetKdTree(this->classic_submap_kdtree_);
    if (this->classic_engine_.needsCovariances())
      this->classic_engine_.setTargetCovariances(this->classic_submap_normals_);
    this->classic_submap_hasChanged_ = false;
  }

  this->classic_engine_.setInputSource(scan_in_map);
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->classic_engine_.align(*aligned);

  Eigen::Matrix4f T_corr = this->classic_engine_.getFinalTransformation();
  score = this->calc_scan_score();
  converged = this->classic_engine_.hasConverged();

  Eigen::Matrix4f T_meas = T_corr * T_predicted;
  meas_p = T_meas.block<3, 1>(0, 3);
  meas_q = Eigen::Quaternionf(Eigen::Matrix3f(T_meas.block<3, 3>(0, 0)));
  meas_q.normalize();
  return true;
}

double c_lio::OdomNode::calc_scan_score()
{
  // Fitness scoring — the existing/old function, unchanged.
  return this->classic_engine_.getFitnessScore(1.0);
}

bool c_lio::OdomNode::apply_corr_gate(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q,
                                      double score, bool converged)
{
  if (this->classic_tick_count_.load() % 40 == 1)
  {
    RCLCPP_DEBUG(this->get_logger(),
                 "[classic] gate input: meas=[%.2f, %.2f, %.2f] score=%.4f (max=%.4f) converged=%d",
                 meas_p[0], meas_p[1], meas_p[2], score,
                 this->classic_max_corr_scan_score_threshold_, static_cast<int>(converged));
  }

  // Mirrors evaluatePoseGate()'s gate 0/1/3/4 (odom_registration.cc), just
  // against classic_kf_'s current pose instead of state/lidarPose.
  if (!converged)
    return false;
  if (!std::isfinite(score) || score > this->classic_max_corr_scan_score_threshold_)
    return false;

  Eigen::Vector3f cur_p;
  Eigen::Quaternionf cur_q;
  {
    std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
    cur_p = this->classic_kf_.position();
    cur_q = this->classic_kf_.orientation();
  }

  float trans = (meas_p - cur_p).norm();
  if (trans > this->classic_max_corr_translation_)
    return false;

  Eigen::Quaternionf dq = cur_q.conjugate() * meas_q;
  double rot_deg = 2.0 * std::acos(std::min(1.0f, std::abs(dq.w()))) * 180.0 / M_PI;
  if (rot_deg > this->classic_max_corr_rotation_deg_)
    return false;

  return true;
}

void c_lio::OdomNode::fuse_odom(const Eigen::Vector3f &meas_p, const Eigen::Quaternionf &meas_q)
{
  std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
  this->classic_kf_.update(meas_p, meas_q);
}

void c_lio::OdomNode::seedClassicKF()
{
  // Seed from classic_seed_p_/classic_seed_q_ (config default at startup, or
  // the last /initialpose click's map-frame guess). Deliberately NOT
  // initial_position_/state.q — those belong to the OLD keyframe pipeline;
  // reading them here would mean an /initialpose click while classic mode is
  // active perturbs the old pipeline's own state (see classic_seed_p_'s
  // comment in odom.h).
  Eigen::Vector3f p0;
  Eigen::Quaternionf q0;

  std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
  p0 = this->classic_seed_p_;
  q0 = this->classic_seed_q_;
  q0.normalize();

  this->classic_kf_.init(this->classic_kf_params_, p0, q0);
  this->classic_kf_initialized_ = true;
}

void c_lio::OdomNode::classic_localization_routine()
{
  // Unconditional — increments on EVERY invocation, before any guard/return,
  // so publishPose() (a separate, already-working callback group/timer) can
  // report the tick count independently. See the comment on
  // classic_tick_count_'s declaration in odom.h.
  this->classic_tick_count_.fetch_add(1);

  // Defensive: the timer is created based on submap_method_param_ at
  // construction time, but the EFFECTIVE submap_method_ gets sanitized to
  // "keyframe" whenever map_mode_=="mapping" (classic is localization-only —
  // see odom.cc's startup sanitize / applyModeTransition()). map_mode_ can
  // change at runtime via SetMode, so re-check the effective value every
  // tick rather than relying on the timer's one-time creation gate.
  if (this->submap_method_ != "classic")
  {
    // Gated on the tick counter (not RCLCPP_WARN_THROTTLE/this->get_clock()) —
    // this->now()/get_clock()->now() were observed frozen in at least one
    // deployment environment (use_sim_time without a working /clock source),
    // which silently breaks ALL clock-based throttling, including this.
    if (this->classic_tick_count_.load() % 100 == 1)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[classic] routine ticking but submap_method_=='%s' (not \"classic\") — "
                  "no-op this tick. If odom/submap/method IS set to \"classic\" in your config, "
                  "check map/mode: classic is sanitized to \"keyframe\" whenever map_mode_==\"mapping\".",
                  this->submap_method_.c_str());
    }
    return;
  }

  // "Set initial pose" (RViz "2D Pose Estimate"): drain the same
  // request_fresh_reloc_ flag callbackInitialPose() sets, and snap
  // classic_kf_ directly to the click — no ScanContext+GICP relocalization
  // search (classic mode doesn't use it; its own per-tick frozen-map
  // registration takes over immediately, which is already its normal job).
  if (this->request_fresh_reloc_.exchange(false))
  {
    Eigen::Vector3f seed_p_log;
    {
      std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
      seed_p_log = this->classic_seed_p_;
    }
    RCLCPP_INFO(this->get_logger(),
                "[classic] /initialpose received — snapping to [%.1f, %.1f]",
                seed_p_log[0], seed_p_log[1]);
    this->classic_roi_initialized_ = false;
    this->seedClassicKF();

    auto [p, q, seq] = this->update_unlocalized_odom();
    this->cls_prev_unlocalized_odom_p_ = p;
    this->cls_prev_unlocalized_odom_q_ = q;
    this->classic_last_source_seq_used_ = seq;
  }

  if (!this->prior_map_cloud_ || this->prior_map_cloud_->empty() || !this->prior_map_kdtree_)
  {
    if (this->classic_tick_count_.load() % 40 == 1)
    {
      RCLCPP_WARN(this->get_logger(), "[classic] prior map not loaded — cannot run classic localization");
    }
    return;
  }

  if (!this->classic_kf_initialized_)
  {
    this->seedClassicKF();
    auto [p, q, seq] = this->update_unlocalized_odom();
    this->cls_prev_unlocalized_odom_p_ = p;
    this->cls_prev_unlocalized_odom_q_ = q;
    this->classic_last_source_seq_used_ = seq;
    RCLCPP_INFO(this->get_logger(),
                "[classic] bootstrapped: seeded at [%.2f, %.2f, %.2f] — will start "
                "predict+register+fuse from the next tick",
                this->classic_kf_.position()[0], this->classic_kf_.position()[1],
                this->classic_kf_.position()[2]);
    return;
  }

  // ---- Skip this tick entirely if the source hasn't produced anything new
  // yet (plain sequence-number comparison — NOT clock-based; see
  // latest_scan_seq_'s comment in odom.h for why). classic_localization_
  // timer_ runs at its own configured rate, decoupled from the scan/topic
  // rate — without this check, a faster timer would repeatedly compute a
  // zero delta from the same stale value between real updates, then jump
  // once new data lands: a stair-step pattern that reads as jitter, even
  // though it has nothing to do with predict/fuse math. ----
  auto [p, q, src_seq] = this->update_unlocalized_odom();
  if (src_seq <= this->classic_last_source_seq_used_)
  {
    return;
  }
  this->classic_last_source_seq_used_ = src_seq;

  // ---- Predict: delta = prev_unlocalized_odom^-1 * unlocalized_odom ----
  Eigen::Isometry3d prev_iso = Eigen::Isometry3d::Identity();
  prev_iso.linear() = this->cls_prev_unlocalized_odom_q_.toRotationMatrix().cast<double>();
  prev_iso.translation() = this->cls_prev_unlocalized_odom_p_.cast<double>();

  Eigen::Isometry3d curr_iso = Eigen::Isometry3d::Identity();
  curr_iso.linear() = q.toRotationMatrix().cast<double>();
  curr_iso.translation() = p.cast<double>();

  Eigen::Isometry3d delta = prev_iso.inverse() * curr_iso;
  Eigen::Vector3f delta_p = delta.translation().cast<float>();
  Eigen::Quaternionf delta_q(Eigen::Matrix3f(delta.rotation().cast<float>()));
  delta_q.normalize();

  {
    std::lock_guard<std::mutex> lock(this->classic_state_mtx_);
    this->classic_kf_.predict(delta_p, delta_q);
  }

  this->cls_prev_unlocalized_odom_p_ = p;
  this->cls_prev_unlocalized_odom_q_ = q;

  // ---- Register + gate + fuse ----
  Eigen::Vector3f meas_p;
  Eigen::Quaternionf meas_q;
  double score = 0.0;
  bool converged = false;
  if (!this->registration_to_prior_map(meas_p, meas_q, score, converged))
  {
    if (this->classic_tick_count_.load() % 40 == 1)
    {
      RCLCPP_WARN(this->get_logger(),
                  "[classic] registration_to_prior_map() returned false this tick "
                  "(likely latest_scan_ not populated yet — check the scan callback's "
                  "latest_scan_ gate condition, odom_callbacks.cc) — predict-only");
    }
    return;
  }

  if (this->apply_corr_gate(meas_p, meas_q, score, converged))
  {
    this->fuse_odom(meas_p, meas_q);
  }
  else if (this->classic_tick_count_.load() % 40 == 1)
  {
    RCLCPP_WARN(this->get_logger(),
                "[classic] correction rejected (score=%.4f converged=%d) — predict-only this tick",
                score, static_cast<int>(converged));
  }
}
