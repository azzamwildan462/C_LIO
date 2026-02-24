#include "dlio/odom.h"
#include "dlio/utils.h"
#include "dlio/kfdb_io.h"

#include <filesystem>

void dlio::OdomNode::loadPriorMap()
{
  // Resolve load path: try corrected file first if use_corrected is enabled
  std::string load_path = this->map_path_;
  if (this->use_corrected_)
  {
    std::filesystem::path p(this->map_path_);
    std::string corrected = p.parent_path().string() + "/" + p.stem().string() + "_corrected.pcd";
    if (std::filesystem::exists(corrected))
    {
      load_path = corrected;
      RCLCPP_INFO(this->get_logger(), "Using corrected map: %s", corrected.c_str());
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Corrected map not found (%s), falling back to raw", corrected.c_str());
    }
  }

  RCLCPP_INFO(this->get_logger(), "Loading prior map from: %s", load_path.c_str());

  // 1. Load PCD file
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();
  if (pcl::io::loadPCDFile(load_path, *cloud) == -1)
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", load_path.c_str());
    this->use_prior_map_ = false;
    return;
  }

  if (cloud->points.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Loaded PCD file is empty!");
    this->use_prior_map_ = false;
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Loaded %zu points from PCD", cloud->points.size());

  // 2. Voxel filter
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(this->map_voxel_size_, this->map_voxel_size_, this->map_voxel_size_);
  vg.setInputCloud(cloud);
  vg.filter(*cloud);

  RCLCPP_INFO(this->get_logger(), "After voxel filter: %zu points", cloud->points.size());

  // Always store prior map cloud + KdTree for potential relocalization.
  // SC database will be built lazily on first relocalization attempt,
  // because ComposableNode parameter overrides (relocalize, map_mode)
  // may not be resolved correctly during constructor.
  this->prior_map_cloud_ = std::make_shared<pcl::PointCloud<PointType>>(*cloud);
  this->prior_map_kdtree_ = std::make_shared<nanoflann::KdTreeFLANN<PointType>>();
  this->prior_map_kdtree_->setInputCloud(this->prior_map_cloud_);
  RCLCPP_INFO(this->get_logger(), "Prior map cloud stored: %zu pts, KdTree built",
              this->prior_map_cloud_->size());

  // 3. Compute covariances for all points (GICP only)
  std::shared_ptr<const nano_gicp::CovarianceList> all_covs;
  if (this->use_gicp_)
  {
    nano_gicp::NanoGICP<PointType, PointType> tmp_gicp;
    tmp_gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
    tmp_gicp.setInputSource(cloud);
    tmp_gicp.calculateSourceCovariances();
    all_covs = tmp_gicp.getSourceCovariances();
  }

  // 4. Split into spatial grid chunks
  double cs = this->map_chunk_size_;
  std::map<std::pair<int, int>, std::vector<int>> grid;

  for (size_t i = 0; i < cloud->points.size(); i++)
  {
    int gx = static_cast<int>(std::floor(cloud->points[i].x / cs));
    int gy = static_cast<int>(std::floor(cloud->points[i].y / cs));
    grid[{gx, gy}].push_back(static_cast<int>(i));
  }

  // 5. Create virtual keyframes from each chunk
  int min_points_per_chunk = 50;
  rclcpp::Time placeholder_stamp(0, 0, RCL_ROS_TIME);
  Eigen::Quaternionf identity_q(1.f, 0.f, 0.f, 0.f);
  Eigen::Matrix4f identity_T = Eigen::Matrix4f::Identity();

  for (auto &[cell, indices] : grid)
  {
    if (static_cast<int>(indices.size()) < min_points_per_chunk)
      continue;

    // Build chunk cloud and covariances
    pcl::PointCloud<PointType>::Ptr chunk_cloud = std::make_shared<pcl::PointCloud<PointType>>();
    chunk_cloud->points.resize(indices.size());
    std::shared_ptr<nano_gicp::CovarianceList> chunk_covs;
    if (this->use_gicp_)
    {
      chunk_covs = std::make_shared<nano_gicp::CovarianceList>(indices.size());
    }

    Eigen::Vector3f centroid(0.f, 0.f, 0.f);
    for (size_t j = 0; j < indices.size(); j++)
    {
      chunk_cloud->points[j] = cloud->points[indices[j]];
      if (this->use_gicp_)
      {
        (*chunk_covs)[j] = (*all_covs)[indices[j]];
      }
      centroid += chunk_cloud->points[j].getVector3fMap();
    }
    centroid /= static_cast<float>(indices.size());
    chunk_cloud->width = chunk_cloud->points.size();
    chunk_cloud->height = 1;
    chunk_cloud->is_dense = true;

    // Add as keyframe (cloud is already in world frame)
    this->keyframes.push_back(std::make_pair(
        std::make_pair(centroid, identity_q),
        pcl::PointCloud<PointType>::ConstPtr(chunk_cloud)));
    this->keyframe_timestamps.push_back(placeholder_stamp);
    this->keyframe_normals.push_back(chunk_covs);
    this->keyframe_transformations.push_back(identity_T);
  }

  // 6. Mark all loaded keyframes as already processed (they're in world frame)
  this->num_processed_keyframes = this->keyframes.size();
  this->num_prior_keyframes_ = this->keyframes.size();

  // 6b. Load KFDB (real keyframe SC descriptors) if available, else fall back to chunk-based SC
  if (this->loadKeyframeDatabase())
  {
    RCLCPP_INFO(this->get_logger(), "ScanContext: using %zu real keyframe descriptors from KFDB",
                this->sc_database_.size());
  }
  else if (this->prior_map_cloud_ && !this->keyframes.empty())
  {
    this->buildScanContextDatabase();
    RCLCPP_INFO(this->get_logger(), "ScanContext: built database with %zu entries from %d map chunks (no KFDB found)",
                this->sc_database_.size(), this->num_prior_keyframes_);
  }

  // 7. Set initial state
  this->state.p = this->initial_position_;
  this->origin = this->initial_position_;

  // 8. Build initial submap synchronously
  this->buildSubmap(this->state);
  this->new_submap_is_ready = true;

  RCLCPP_INFO(this->get_logger(),
              "Prior map loaded: %d chunks (%zu total points), initial pose=[%.1f, %.1f, %.1f]",
              this->num_prior_keyframes_, cloud->points.size(),
              this->initial_position_[0], this->initial_position_[1], this->initial_position_[2]);
}

// SC computation functions moved to dlio/scan_context.h

void dlio::OdomNode::buildScanContextDatabase()
{
  this->sc_database_.clear();

  if (!this->prior_map_cloud_ || this->prior_map_cloud_->empty())
  {
    RCLCPP_WARN(this->get_logger(), "SC build: prior_map_cloud is null or empty!");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "SC build: %zu keyframes, %zu prior map points",
              this->keyframes.size(), this->prior_map_cloud_->size());

  // Use this->sc_max_range_ as search radius so descriptor covers full range
  // nanoflann radiusSearch uses squared L2 distance internally
  float search_radius_sq = this->sc_max_range_ * this->sc_max_range_;

  int skipped = 0;

  // Build one SC entry per keyframe (chunk centroid)
  for (size_t ki = 0; ki < this->keyframes.size(); ki++)
  {
    const auto &kf = this->keyframes[ki];
    Eigen::Vector3f centroid = kf.first.first;

    // Radius search around centroid to get local points
    PointType center_pt;
    center_pt.x = centroid[0];
    center_pt.y = centroid[1];
    center_pt.z = centroid[2];

    std::vector<int> indices;
    std::vector<float> dists;
    this->prior_map_kdtree_->radiusSearch(center_pt, search_radius_sq, indices, dists);

    if (ki < 3 || indices.size() < 50)
    {
      RCLCPP_INFO(this->get_logger(), "SC build: kf[%zu] centroid=[%.1f,%.1f,%.1f] radius_sq=%.0f found=%zu",
                  ki, centroid[0], centroid[1], centroid[2], search_radius_sq, indices.size());
    }

    if (indices.size() < 50)
    {
      skipped++;
      continue;
    }

    // Find min Z in the local region as ground reference
    float z_min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < indices.size(); i++)
    {
      float z = this->prior_map_cloud_->points[indices[i]].z;
      if (z < z_min)
        z_min = z;
    }

    // Re-center XY relative to centroid, Z relative to ground
    // This simulates a sensor-frame view (like an incoming LiDAR scan)
    pcl::PointCloud<PointType>::Ptr local_cloud = std::make_shared<pcl::PointCloud<PointType>>();
    local_cloud->points.resize(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
    {
      local_cloud->points[i].x = this->prior_map_cloud_->points[indices[i]].x - centroid[0];
      local_cloud->points[i].y = this->prior_map_cloud_->points[indices[i]].y - centroid[1];
      local_cloud->points[i].z = this->prior_map_cloud_->points[indices[i]].z - z_min;
    }

    dlio::sc::ScanContextEntry entry;
    entry.descriptor = dlio::sc::computeScanContext(local_cloud, this->sc_max_range_);
    entry.ring_key = dlio::sc::computeRingKey(entry.descriptor);
    entry.sector_key = dlio::sc::computeSectorKey(entry.descriptor);
    entry.position = centroid;
    entry.orientation = Eigen::Quaternionf::Identity();
    this->sc_database_.push_back(std::move(entry));
  }

  RCLCPP_INFO(this->get_logger(), "SC build done: %zu entries, %d skipped (<50 pts in radius)",
              this->sc_database_.size(), skipped);
}

bool dlio::OdomNode::runRelocalization(pcl::PointCloud<PointType>::ConstPtr scan)
{
  // Lazy build: if database wasn't built during constructor (param override timing),
  // try KFDB first, then fall back to chunk-based SC
  if (this->sc_database_.empty() && this->prior_map_cloud_ && !this->keyframes.empty())
  {
    if (!this->loadKeyframeDatabase())
    {
      RCLCPP_INFO(this->get_logger(), "ScanContext: building database lazily from map chunks (%zu keyframes, %zu map pts)",
                  this->keyframes.size(), this->prior_map_cloud_->size());
      this->buildScanContextDatabase();
    }
    RCLCPP_INFO(this->get_logger(), "ScanContext: lazy build done — %zu entries", this->sc_database_.size());
  }

  if (this->sc_database_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "ScanContext: database is empty, cannot relocalize"
                                    " (prior_map=%s, keyframes=%zu)",
                this->prior_map_cloud_ ? "yes" : "no", this->keyframes.size());
    return false;
  }

  // ---- Use original_scan (sensor frame) to avoid deskewing artifacts ----
  // current_scan is in world frame (via deskewing), which causes two problems:
  // 1. GICP init_guess applies gravity rotation again (double gravity alignment)
  // 2. IMU drift between attempts shifts the scan center, degrading SC descriptors
  // By using original_scan in sensor frame, we get consistent behavior across all attempts.

  // Voxel filter the raw scan for efficiency
  pcl::PointCloud<PointType>::Ptr raw_scan = std::make_shared<pcl::PointCloud<PointType>>(*this->original_scan);
  if (this->vf_use_)
  {
    pcl::VoxelGrid<PointType> vf;
    vf.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);
    vf.setInputCloud(raw_scan);
    vf.filter(*raw_scan);
  }

  // 1. Rotate scan to gravity-aligned frame for SC matching
  RCLCPP_INFO(this->get_logger(),
              "SC: using kfdb_gravity_q=[%.4f,%.4f,%.4f,%.4f] ground_thresh=%.2f",
              this->kfdb_gravity_q_.w(), this->kfdb_gravity_q_.x(),
              this->kfdb_gravity_q_.y(), this->kfdb_gravity_q_.z(),
              this->sc_ground_height_threshold_);
  auto sc_scan = dlio::sc::prepareGravityAlignedScan(
      raw_scan, this->kfdb_gravity_q_, this->extrinsics.baselink2lidar.R,
      this->sc_ground_height_threshold_);

  dlio::sc::ScanContextDescriptor query_desc = dlio::sc::computeScanContext(sc_scan, this->sc_max_range_);
  dlio::sc::RingKey query_key = dlio::sc::computeRingKey(query_desc);
  dlio::sc::SectorKey query_sector_key = dlio::sc::computeSectorKey(query_desc);

  // Debug: SC descriptor stats
  {
    int nonzero = 0;
    float max_val = 0.f;
    for (int r = 0; r < dlio::sc::SC_NR; r++)
      for (int s = 0; s < dlio::sc::SC_NS; s++)
      {
        if (query_desc(r, s) != 0.f)
          nonzero++;
        if (query_desc(r, s) > max_val)
          max_val = query_desc(r, s);
      }
    RCLCPP_INFO(this->get_logger(),
                "SC debug: orig_pts=%zu, filtered=%zu, max_range=%.1f, "
                "query_nonzero=%d/%d max_z=%.2f | gravity_q=[%.3f,%.3f,%.3f,%.3f]",
                this->original_scan ? this->original_scan->size() : 0,
                raw_scan->size(), this->sc_max_range_,
                nonzero, dlio::sc::SC_NR * dlio::sc::SC_NS, max_val,
                this->kfdb_gravity_q_.w(), this->kfdb_gravity_q_.x(),
                this->kfdb_gravity_q_.y(), this->kfdb_gravity_q_.z());
    // Print stats for a few database entries
    for (int di : {0, 1, 50, 100, 200, 399})
    {
      if (di >= static_cast<int>(this->sc_database_.size()))
        break;
      const auto &dbd = this->sc_database_[di].descriptor;
      const auto &dbp = this->sc_database_[di].position;
      int dn = 0;
      float dm = 0.f;
      for (int r = 0; r < dlio::sc::SC_NR; r++)
        for (int s = 0; s < dlio::sc::SC_NS; s++)
        {
          if (dbd(r, s) != 0.f)
            dn++;
          if (dbd(r, s) > dm)
            dm = dbd(r, s);
        }
      RCLCPP_INFO(this->get_logger(),
                  "SC db[%d]: nonzero=%d max_z=%.2f pos=[%.1f,%.1f,%.1f]",
                  di, dn, dm, dbp[0], dbp[1], dbp[2]);
    }
  }

  // 2. Compute SC distance for ALL database entries and rank them
  //    SC from map chunks is approximate, so we use it for ranking only (no hard threshold)
  struct SCCandidate
  {
    int db_idx;
    float sc_dist;
    int shift;
  };
  std::vector<SCCandidate> sc_candidates;

  for (size_t i = 0; i < this->sc_database_.size(); i++)
  {
    auto [dist, shift] = dlio::sc::computeScanContextDistance(
        query_desc, this->sc_database_[i].descriptor,
        query_sector_key, this->sc_database_[i].sector_key,
        this->sc_search_window_);
    sc_candidates.push_back({static_cast<int>(i), dist, shift});
  }

  // Sort by SC distance (best first)
  std::sort(sc_candidates.begin(), sc_candidates.end(),
            [](const SCCandidate &a, const SCCandidate &b)
            { return a.sc_dist < b.sc_dist; });

  // Limit to top-K candidates
  int num_candidates = std::min(this->sc_num_candidates_,
                                static_cast<int>(sc_candidates.size()));

  RCLCPP_INFO(this->get_logger(),
              "ScanContext: trying top %d/%zu candidates (best sc_dist=%.3f, worst=%.3f)",
              num_candidates, sc_candidates.size(),
              sc_candidates.front().sc_dist, sc_candidates[num_candidates - 1].sc_dist);

  // 3. Pre-compute source covariances on raw_scan (sensor frame) for GICP
  std::shared_ptr<const nano_gicp::CovarianceList> source_covs;
  if (this->use_gicp_)
  {
    nano_gicp::NanoGICP<PointType, PointType> src_gicp;
    src_gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
    src_gicp.setInputSource(raw_scan);
    src_gicp.calculateSourceCovariances();
    source_covs = src_gicp.getSourceCovariances();
  }

  float refine_radius_sq = 50.f * 50.f;

  // 4. Build list of candidate positions: initial_position first, then SC candidates
  struct GICPCandidate
  {
    Eigen::Vector3f position;
    float sc_yaw; // base yaw from SC (0 for initial_position)
  };
  std::vector<GICPCandidate> positions;

  // Always try initial_position first (configured by user, likely correct)
  positions.push_back({this->initial_position_, 0.f});

  for (int ci = 0; ci < num_candidates; ci++)
  {
    const auto &cand = sc_candidates[ci];
    Eigen::Vector3f pos = this->sc_database_[cand.db_idx].position;
    float yaw = static_cast<float>(cand.shift) * 2.f * M_PI / static_cast<float>(dlio::sc::SC_NS);
    positions.push_back({pos, yaw});
  }

  RCLCPP_INFO(this->get_logger(),
              "ScanContext: trying %zu positions (initial_pose + %d SC candidates)",
              positions.size(), num_candidates);

  // 5. Track global best result across all candidates and yaw hypotheses
  float best_fitness = std::numeric_limits<float>::max();
  Eigen::Vector3f best_pos;
  Eigen::Quaternionf best_q;
  int best_candidate = -1;

  // Precompute baselink2lidar transform and its inverse
  // baselink2lidar_T transforms: lidar frame → baselink frame
  Eigen::Matrix4f B2L_T = this->extrinsics.baselink2lidar_T;
  Eigen::Matrix4f L2B_T = B2L_T.inverse(); // baselink frame → lidar frame

  // Yaw hypotheses: base yaw + {0, 60, 120, 180, 240, 300} degrees
  const float yaw_offsets[] = {0.f, M_PI / 3.f, 2.f * M_PI / 3.f, M_PI, -2.f * M_PI / 3.f, -M_PI / 3.f};

  for (size_t ci = 0; ci < positions.size(); ci++)
  {
    Eigen::Vector3f matched_pos = positions[ci].position;
    float sc_yaw = positions[ci].sc_yaw;

    // Extract local map around candidate (once per position)
    PointType search_pt;
    search_pt.x = matched_pos[0];
    search_pt.y = matched_pos[1];
    search_pt.z = matched_pos[2];

    std::vector<int> local_indices;
    std::vector<float> local_dists;
    this->prior_map_kdtree_->radiusSearch(search_pt, refine_radius_sq, local_indices, local_dists);

    if (local_indices.size() < 100)
      continue;

    pcl::PointCloud<PointType>::Ptr local_map = std::make_shared<pcl::PointCloud<PointType>>();
    local_map->points.resize(local_indices.size());
    for (size_t i = 0; i < local_indices.size(); i++)
    {
      local_map->points[i] = this->prior_map_cloud_->points[local_indices[i]];
    }
    local_map->width = local_map->points.size();
    local_map->height = 1;
    local_map->is_dense = true;

    // Downsample local map to avoid GICP issues with overly dense targets
    size_t local_map_raw_size = local_map->points.size();
    {
      pcl::VoxelGrid<PointType> vf;
      vf.setLeafSize(0.5f, 0.5f, 0.5f);
      vf.setInputCloud(local_map);
      vf.filter(*local_map);
      local_map->width = local_map->points.size();
      local_map->height = 1;
    }

    if (ci == 0)
    {
      RCLCPP_INFO(this->get_logger(),
                  "  GICP debug: raw_scan=%zu pts, local_map=%zu->%zu pts (r=50m around [%.1f,%.1f,%.1f])",
                  raw_scan->size(), local_map_raw_size, local_map->points.size(),
                  matched_pos[0], matched_pos[1], matched_pos[2]);
    }

    // Try multiple yaw hypotheses — fresh GICP per alignment to avoid state corruption
    for (float yaw_offset : yaw_offsets)
    {
      float yaw = sc_yaw + yaw_offset;

      // Build init_guess = world_T_lidar = world_T_baselink * baselink_T_lidar
      // raw_scan is in lidar frame; init_guess transforms it to world frame
      // Note: use state.q (map frame orientation) for GICP, NOT gravity_q (SC frame only)
      Eigen::Matrix4f world_T_baselink = Eigen::Matrix4f::Identity();
      Eigen::Quaternionf yaw_q(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
      Eigen::Quaternionf init_q = yaw_q * this->state.q;
      world_T_baselink.block<3, 3>(0, 0) = init_q.toRotationMatrix();
      world_T_baselink.block<3, 1>(0, 3) = matched_pos;
      Eigen::Matrix4f init_guess = world_T_baselink * B2L_T;

      // Fresh registration object per alignment to prevent internal state corruption
      pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
      bool converged = false;
      float fitness = std::numeric_limits<float>::max();
      Eigen::Matrix4f T_final = Eigen::Matrix4f::Identity();

      if (this->use_gicp_)
      {
        nano_gicp::NanoGICP<PointType, PointType> gicp;
        gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
        gicp.setMaxCorrespondenceDistance(5.0);
        gicp.setMaximumIterations(64);
        gicp.setTransformationEpsilon(0.01);
        gicp.setRotationEpsilon(0.01);

        gicp.setInputSource(raw_scan);
        gicp.setSourceCovariances(source_covs);
        gicp.setInputTarget(local_map);
        gicp.calculateTargetCovariances();

        gicp.align(*aligned, init_guess);
        converged = gicp.hasConverged();
        fitness = gicp.getFitnessScore(1.0);
        T_final = gicp.getFinalTransformation();
      }
      else
      {
        pclomp::NormalDistributionsTransform<PointType, PointType> ndt_local;
        ndt_local.setResolution(this->ndt_resolution_);
        ndt_local.setNumThreads(this->ndt_num_threads_);
        ndt_local.setNeighborhoodSearchMethod(pclomp::DIRECT7);
        ndt_local.setMaximumIterations(64);
        ndt_local.setTransformationEpsilon(0.01);

        ndt_local.setInputSource(raw_scan);
        ndt_local.setInputTarget(local_map);

        ndt_local.align(*aligned, init_guess);
        converged = ndt_local.hasConverged();
        fitness = ndt_local.getFitnessScore(1.0);
        T_final = ndt_local.getFinalTransformation();
      }

      if (!converged)
        continue;

      if (fitness < best_fitness)
      {
        best_fitness = fitness;
        Eigen::Matrix4f W_T_B = T_final * L2B_T;
        best_pos = W_T_B.block<3, 1>(0, 3);
        Eigen::Quaternionf rq(W_T_B.block<3, 3>(0, 0));
        rq.normalize();
        best_q = rq;
        best_candidate = ci;

        RCLCPP_INFO(this->get_logger(),
                    "  new best: pos=[%.1f,%.1f,%.1f] fitness=%.4f (candidate %zu/%zu, yaw_init=%.0f deg)",
                    best_pos[0], best_pos[1], best_pos[2], fitness, ci, positions.size(), yaw * 180.f / M_PI);

        // Early exit if fitness is already excellent
        if (fitness < 0.3f)
          goto accept_result;
      }
    }
  }

accept_result:
  if (best_fitness > 0.5f)
  {
    RCLCPP_WARN(this->get_logger(),
                "ScanContext: best GICP fitness %.4f > 0.5, rejecting (tried %zu positions x %zu yaws)",
                best_fitness, positions.size(), std::size(yaw_offsets));
    return false;
  }

  this->state.p = best_pos;
  this->origin = best_pos;
  this->state.q = best_q;
  this->last_reloc_fitness_ = best_fitness;

  float refined_yaw = std::atan2(
      2.f * (best_q.w() * best_q.z() + best_q.x() * best_q.y()),
      1.f - 2.f * (best_q.y() * best_q.y() + best_q.z() * best_q.z()));

  RCLCPP_INFO(this->get_logger(),
              "ScanContext: relocalized to [%.1f, %.1f, %.1f] yaw=%.1f deg (GICP fitness=%.4f, candidate %d/%zu)",
              best_pos[0], best_pos[1], best_pos[2],
              refined_yaw * 180.f / M_PI, best_fitness, best_candidate + 1, positions.size());

  return true;
}

// ---- Keyframe Database (KFDB) ----

std::string dlio::OdomNode::getKfdbPath() const
{
  return dlio::kfdb::getKfdbPath(this->map_path_, this->use_corrected_);
}

void dlio::OdomNode::computeAndStoreKeyframeSC()
{
  if (this->map_mode_ != "mapping")
    return;
  if (!this->original_scan || this->original_scan->empty())
    return;

  // 1. Voxel filter original_scan
  pcl::PointCloud<PointType>::Ptr raw_scan =
      std::make_shared<pcl::PointCloud<PointType>>(*this->original_scan);
  if (this->vf_use_)
  {
    pcl::VoxelGrid<PointType> vf;
    vf.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);
    vf.setInputCloud(raw_scan);
    vf.filter(*raw_scan);
  }

  // 2. Rotate to gravity-aligned frame for SC consistency (with ground removal)
  auto sc_scan = dlio::sc::prepareGravityAlignedScan(
      raw_scan, this->kfdb_gravity_q_, this->extrinsics.baselink2lidar.R,
      this->sc_ground_height_threshold_);

  // 3. Compute SC descriptor, ring key, and sector key (SC++)
  dlio::sc::ScanContextEntry entry;
  entry.descriptor = dlio::sc::computeScanContext(sc_scan, this->sc_max_range_);
  entry.ring_key = dlio::sc::computeRingKey(entry.descriptor);
  entry.sector_key = dlio::sc::computeSectorKey(entry.descriptor);
  entry.position = this->lidarPose.p;
  entry.orientation = this->lidarPose.q;

  // Attach GPS data to keyframe entry
  GPSMeasurement gps_at_kf;
  if (this->gps_enabled_ && this->getGPSAtTime(this->scan_header_stamp.seconds(), gps_at_kf))
  {
    entry.gps_latitude = gps_at_kf.latitude;
    entry.gps_longitude = gps_at_kf.longitude;
    entry.gps_altitude = gps_at_kf.altitude;
    entry.gps_valid = true;
  }
  else
  {
    entry.gps_valid = false;
  }

  // Debug: log first 3 entries
  {
    std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
    int idx = this->kfdb_entries_.size();
    if (idx < 3)
    {
      int nz = 0;
      float mx = 0.f;
      for (int r = 0; r < dlio::sc::SC_NR; r++)
        for (int s = 0; s < dlio::sc::SC_NS; s++)
        {
          if (entry.descriptor(r, s) != 0.f)
            nz++;
          if (entry.descriptor(r, s) > mx)
            mx = entry.descriptor(r, s);
        }
      RCLCPP_INFO(this->get_logger(),
                  "KFDB save[%d]: orig=%zu filtered=%zu nonzero=%d max_z=%.2f q=[%.4f,%.4f,%.4f,%.4f] pos=[%.2f,%.2f,%.2f]",
                  idx, this->original_scan ? this->original_scan->size() : 0,
                  raw_scan->size(), nz, mx,
                  this->state.q.w(), this->state.q.x(), this->state.q.y(), this->state.q.z(),
                  entry.position[0], entry.position[1], entry.position[2]);
    }
  }

  // 5. Store and periodically save to disk
  {
    std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
    this->kfdb_entries_.push_back(std::move(entry));
  }

  // Auto-save KFDB every 20 keyframes (cheap — a few KB per entry)
  if (this->kfdb_entries_.size() % 20 == 0)
  {
    this->saveKeyframeDatabase();
  }
}

bool dlio::OdomNode::saveKeyframeDatabase()
{
  std::string kfdb_path = this->getKfdbPath();
  if (kfdb_path.empty())
    return false;

  // Brief lock: copy entries
  std::vector<dlio::sc::ScanContextEntry> entries_snap;
  float max_range;
  Eigen::Quaternionf gravity_q;
  {
    std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
    if (this->kfdb_entries_.empty())
    {
      RCLCPP_INFO(this->get_logger(), "KFDB: no entries to save");
      return false;
    }
    entries_snap = this->kfdb_entries_;
    max_range = this->sc_max_range_;
    gravity_q = this->kfdb_gravity_q_;
  }

  // File I/O outside lock
  bool ok = dlio::kfdb::save(kfdb_path, entries_snap, max_range, gravity_q);
  if (ok)
    RCLCPP_INFO(this->get_logger(), "KFDB: saved %zu entries to %s",
                entries_snap.size(), kfdb_path.c_str());
  else
    RCLCPP_ERROR(this->get_logger(), "KFDB: failed to save to %s", kfdb_path.c_str());
  return ok;
}

bool dlio::OdomNode::saveCorrectedKeyframeDatabase()
{
  std::vector<geometry_msgs::msg::Pose> corrected_poses;
  {
    std::lock_guard<std::mutex> lock(this->corrected_kf_poses_mutex_);
    corrected_poses = this->corrected_kf_poses_;
  }

  if (corrected_poses.empty())
  {
    RCLCPP_INFO(this->get_logger(), "KFDB corrected: no corrected poses from graph_slam, skipping");
    return false;
  }

  std::lock_guard<std::mutex> lock(this->kfdb_mutex_);
  if (this->kfdb_entries_.empty())
    return false;

  bool ok = dlio::kfdb::saveCorrected(this->map_path_, this->kfdb_entries_,
                                      corrected_poses, this->sc_max_range_,
                                      this->kfdb_gravity_q_);
  if (ok)
    RCLCPP_INFO(this->get_logger(), "KFDB corrected: saved to %s", this->map_path_.c_str());
  else
    RCLCPP_ERROR(this->get_logger(), "KFDB corrected: save failed");
  return ok;
}

bool dlio::OdomNode::loadKeyframeDatabase()
{
  std::string kfdb_path = this->getKfdbPath();
  if (kfdb_path.empty())
    return false;

  float file_max_range = this->sc_max_range_;
  Eigen::Quaternionf file_gravity_q = this->kfdb_gravity_q_;

  bool ok = dlio::kfdb::load(kfdb_path, this->sc_database_,
                             file_max_range, file_gravity_q);
  if (!ok)
    return false;

  // Update members from file header
  this->kfdb_gravity_q_ = file_gravity_q;
  if (std::abs(file_max_range - this->sc_max_range_) > 0.1f)
  {
    RCLCPP_WARN(this->get_logger(), "KFDB: sc_max_range mismatch (file: %.1f, param: %.1f), using file's value",
                file_max_range, this->sc_max_range_);
    this->sc_max_range_ = file_max_range;
  }

  RCLCPP_INFO(this->get_logger(), "KFDB: loaded %zu entries from %s (sc_max_range=%.1f, gravity_q=[%.4f,%.4f,%.4f,%.4f])",
              this->sc_database_.size(), kfdb_path.c_str(), this->sc_max_range_,
              this->kfdb_gravity_q_.w(), this->kfdb_gravity_q_.x(),
              this->kfdb_gravity_q_.y(), this->kfdb_gravity_q_.z());
  return true;
}
