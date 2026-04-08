#include "dlio/odom.h"
#include "dlio/utils.h"

#include <queue>

void dlio::OdomNode::computeMetrics(pcl::PointCloud<PointType>::ConstPtr scan)
{
  this->computeSpaciousness(scan);
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness(pcl::PointCloud<PointType>::ConstPtr scan)
{

  // compute range of points
  std::vector<float> ds;

  for (int i = 0; i < scan->points.size(); i++)
  {
    float d = std::sqrt(pow(scan->points[i].x, 2) +
                        pow(scan->points[i].y, 2));
    ds.push_back(d);
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
  float median_curr = ds[ds.size() / 2];
  float median_lpf = 0.95 * this->spaciousness_median_prev_ + 0.05 * median_curr;
  this->spaciousness_median_prev_ = median_lpf;

  // push (lock to prevent race with setAdaptiveParams reading .back())
  {
    std::lock_guard<std::mutex> lock(this->metrics_mtx_);
    this->metrics.spaciousness.push_back(median_lpf);
  }
}

void dlio::OdomNode::computeDensity()
{

  float density;

  if (!this->geo.first_opt_done)
  {
    density = 0.;
  }
  else if (this->use_gicp_)
  {
    density = this->gicp.source_density_;
  }
  else
  {
    density = this->gicp_max_corr_dist_; // NDT has no density metric
  }

  float density_lpf = 0.95 * this->density_prev_ + 0.05 * density;
  this->density_prev_ = density_lpf;

  {
    std::lock_guard<std::mutex> lock(this->metrics_mtx_);
    this->metrics.density.push_back(density_lpf);
  }
}

void dlio::OdomNode::computeConvexHull()
{

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4)
  {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++)
  {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i = 0; i < convex_hull_point_idx->indices.size(); ++i)
  {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }
}

void dlio::OdomNode::computeConcaveHull()
{

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5)
  {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++)
  {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i = 0; i < concave_hull_point_idx->indices.size(); ++i)
  {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }
}

void dlio::OdomNode::updateKeyframes()
{

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto &k : this->keyframes)
  {

    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt(pow(this->state.p[0] - k.first.first[0], 2) +
                         pow(this->state.p[1] - k.first.first[1], 2) +
                         pow(this->state.p[2] - k.first.first[2], 2));

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5)
    {
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d)
    {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;
  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt(pow(this->state.p[0] - closest_pose[0], 2) +
                  pow(this->state.p[1] - closest_pose[1], 2) +
                  pow(this->state.p[2] - closest_pose[2], 2));

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.)
  {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.;
    lq.x() *= -1.;
    lq.y() *= -1.;
    lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  }
  else
  {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt(pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2)), dq.w());
  double theta_deg = theta_rad * (180.0 / M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_)
  {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_)
  {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1)
  {
    newKeyframe = true;
  }

  if (newKeyframe)
  {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
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
    lock.unlock();

    // Compute and store SC descriptor for this keyframe (mapping mode)
    this->computeAndStoreKeyframeSC();
  }
}

void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames)
{

  // make sure dists is not empty
  if (!dists.size())
  {
    return;
  }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists)
  {
    if (pq.size() >= k && pq.top() > d)
    {
      pq.push(d);
      pq.pop();
    }
    else if (pq.size() < k)
    {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i)
  {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }
}

void dlio::OdomNode::buildSubmap(State vehicle_state)
{

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++)
  {
    float d = sqrt(pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                   pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                   pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2));
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto &c : this->keyframe_convex)
  {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto &c : this->keyframe_concave)
  {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev)
  {

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    if (this->submap_method_ == "voxel_hash_map" && this->voxel_map_)
    {
      // Voxel hash map: incrementally add new keyframes only
      int num_kf = static_cast<int>(this->keyframes.size());
      for (int k = this->voxel_map_last_kf_idx_; k < num_kf; ++k)
      {
        lock.lock();
        auto kf_cloud = this->keyframes[k].second;
        lock.unlock();
        this->voxel_map_->update(kf_cloud, vehicle_state.p);
      }
      this->voxel_map_last_kf_idx_ = num_kf;

      this->submap_cloud = this->voxel_map_->getCloud();
    }
    else
    {
      // KNN keyframe submap (default)
      pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
      std::shared_ptr<nano_gicp::CovarianceList> submap_normals_;
      if (this->use_gicp_)
      {
        submap_normals_ = std::make_shared<nano_gicp::CovarianceList>();
      }

      for (auto k : this->submap_kf_idx_curr)
      {
        lock.lock();
        *submap_cloud_ += *this->keyframes[k].second;
        lock.unlock();

        if (this->use_gicp_)
        {
          submap_normals_->insert(std::end(*submap_normals_),
                                  std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])));
        }
      }

      this->submap_cloud = submap_cloud_;
      if (this->use_gicp_)
      {
        this->submap_normals = submap_normals_;
      }
    }

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    if (this->registration_method_ == "robust_icp")
    {
      this->robust_icp_temp_.setInputTarget(this->submap_cloud);
      this->submap_kdtree = this->robust_icp_temp_.target_kdtree_;
    }
    else if (this->use_gicp_)
    {
      this->gicp_temp.setInputTarget(this->submap_cloud);
      this->submap_kdtree = this->gicp_temp.target_kdtree_;
    }
    else
    {
      this->ndt_temp.setInputTarget(this->submap_cloud);
    }

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state)
{
  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: enter, num_processed=%d, keyframes=%zu",
                this->num_processed_keyframes, this->keyframes.size());

  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++)
  {
    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: processing kf[%d], cloud=%zu pts, normals=%zu",
                  i, this->keyframes[i].second ? this->keyframes[i].second->points.size() : 0,
                  (this->use_gicp_ && i < (int)this->keyframe_normals.size()) ? this->keyframe_normals[i]->size() : 0);

    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*raw_keyframe, *transformed_keyframe, T);

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;

    if (this->use_gicp_)
    {
      if (this->deep_debug_)
        RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: transforming covariances for kf[%d]", i);
      std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
      Eigen::Matrix4d Td = T.cast<double>();
      std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances(std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
      std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                     [&Td](Eigen::Matrix4d cov)
                     { return Td * cov * Td.transpose(); });
      this->keyframe_normals[i] = transformed_covariances;
    }

    // Copy data for publish, then unlock before heavy serialization
    auto kf_data = this->keyframes[i];
    auto kf_ts = this->keyframe_timestamps[i];
    auto kf_cloud = this->keyframes[i].second;
    lock.unlock();

    if (this->deep_debug_)
      RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: publishing kf[%d]", i);

    // Publish in detached thread — don't block submap build
    // (kf_publish_mtx_ inside publishKeyframe protects shared state)
    std::thread(&dlio::OdomNode::publishKeyframe, this, kf_data, kf_ts, kf_cloud).detach();

    lock.lock();
  }

  lock.unlock();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: all kf processed, calling buildSubmap");

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: buildSubmap start");
  this->buildSubmap(vehicle_state);
  if (this->deep_debug_)
    RCLCPP_INFO(this->get_logger(), "[DEEP] buildKF&Submap: buildSubmap done");
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded()
{
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]
                             { return !this->main_loop_running; });
}
