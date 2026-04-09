/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * Yamato Infrastructure Robotics,                         *
 * Japan                                                   *
 *                                                         *
 * Authors: Azzam Wildan M                                 *
 *                                                         *
 ***********************************************************/

#include "dlio/engines/appearance_engine.h"
#include "dlio/odom/scan_context.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dlio
{

  AppearanceMethod parseAppearanceMethod(const std::string &s)
  {
    if (s == "sc++" || s == "SC++" || s == "scan_context")
      return AppearanceMethod::SC_PLUS_PLUS;
    if (s == "sc++_cuda" || s == "SC++_CUDA" || s == "scan_context_cuda")
      return AppearanceMethod::SC_PLUS_PLUS_CUDA;
    if (s == "std" || s == "STD")
      return AppearanceMethod::STD;
    if (s == "std_cuda" || s == "STD_CUDA")
      return AppearanceMethod::STD_CUDA;

    RCLCPP_WARN(rclcpp::get_logger("appearance_engine"),
                "Unknown appearance method '%s', falling back to sc++", s.c_str());
    return AppearanceMethod::SC_PLUS_PLUS;
  }

  std::string appearanceMethodToString(AppearanceMethod m)
  {
    switch (m)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
      return "sc++";
    case AppearanceMethod::SC_PLUS_PLUS_CUDA:
      return "sc++_cuda";
    case AppearanceMethod::STD:
      return "std";
    case AppearanceMethod::STD_CUDA:
      return "std_cuda";
    }
    return "unknown";
  }

  void AppearanceEngine::init(AppearanceMethod method, const AppearanceParams &params,
                              rclcpp::Logger logger)
  {
    method_ = method;
    params_ = params;
    logger_ = logger;
    initialized_ = true;

    // Validate and fall back for unimplemented methods
    switch (method_)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
      break; // supported
    case AppearanceMethod::SC_PLUS_PLUS_CUDA:
      RCLCPP_WARN(logger_, "sc++_cuda not yet implemented, falling back to sc++");
      method_ = AppearanceMethod::SC_PLUS_PLUS;
      break;
    case AppearanceMethod::STD:
      RCLCPP_WARN(logger_, "STD not yet implemented, falling back to sc++");
      method_ = AppearanceMethod::SC_PLUS_PLUS;
      break;
    case AppearanceMethod::STD_CUDA:
      RCLCPP_WARN(logger_, "STD_CUDA not yet implemented, falling back to sc++");
      method_ = AppearanceMethod::SC_PLUS_PLUS;
      break;
    }

    RCLCPP_INFO(logger_, "AppearanceEngine: method=%s, max_range=%.1f, ground_thresh=%.2f, search_window=%d",
                appearanceMethodToString(method_).c_str(),
                params_.max_range, params_.ground_height_threshold, params_.search_window);
  }

  AppearanceDescriptor AppearanceEngine::computeDescriptor(
      pcl::PointCloud<PointType>::ConstPtr cloud)
  {
    AppearanceDescriptor desc;

    switch (method_)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
    case AppearanceMethod::SC_PLUS_PLUS_CUDA: // fallback
    case AppearanceMethod::STD:               // fallback
    case AppearanceMethod::STD_CUDA:          // fallback
    {
      desc.sc_descriptor = dlio::sc::computeScanContext(cloud, params_.max_range);
      desc.ring_key = dlio::sc::computeRingKey(desc.sc_descriptor);
      desc.sector_key = dlio::sc::computeSectorKey(desc.sc_descriptor);
      break;
    }
    }

    return desc;
  }

  std::pair<float, int> AppearanceEngine::compareDescriptors(
      const AppearanceDescriptor &a, const AppearanceDescriptor &b)
  {
    switch (method_)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
    case AppearanceMethod::SC_PLUS_PLUS_CUDA:
    case AppearanceMethod::STD:
    case AppearanceMethod::STD_CUDA:
    {
      return dlio::sc::computeScanContextDistance(
          a.sc_descriptor, b.sc_descriptor,
          a.sector_key, b.sector_key,
          params_.search_window);
    }
    }

    return {std::numeric_limits<float>::max(), 0};
  }

  std::vector<std::pair<int, float>> AppearanceEngine::findCandidates(
      const AppearanceDescriptor &query,
      const std::vector<AppearanceEntry> &database,
      int top_k, float max_dist)
  {
    struct Candidate
    {
      int idx;
      float dist;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(database.size());

    for (size_t i = 0; i < database.size(); i++)
    {
      auto [dist, shift] = compareDescriptors(query, database[i].descriptor);
      if (max_dist > 0.0f && dist > max_dist)
        continue;
      candidates.push_back({static_cast<int>(i), dist});
    }

    // Sort ascending by distance
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b)
              { return a.dist < b.dist; });

    // Take top-K
    int n = (top_k > 0) ? std::min(top_k, static_cast<int>(candidates.size()))
                        : static_cast<int>(candidates.size());

    std::vector<std::pair<int, float>> result;
    result.reserve(n);
    for (int i = 0; i < n; i++)
      result.emplace_back(candidates[i].idx, candidates[i].dist);

    return result;
  }

  pcl::PointCloud<PointType>::Ptr AppearanceEngine::prepareGravityAlignedScan(
      pcl::PointCloud<PointType>::ConstPtr scan,
      const Eigen::Quaternionf &gravity_q,
      const Eigen::Matrix3f &R_body_to_lidar)
  {
    return dlio::sc::prepareGravityAlignedScan(
        scan, gravity_q, R_body_to_lidar, params_.ground_height_threshold);
  }

  float AppearanceEngine::shiftToYaw(int shift) const
  {
    switch (method_)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
    case AppearanceMethod::SC_PLUS_PLUS_CUDA:
      return static_cast<float>(shift) * 2.f * M_PI / static_cast<float>(dlio::sc::SC_NS);
    case AppearanceMethod::STD:
    case AppearanceMethod::STD_CUDA:
      // STD will have its own alignment semantics
      return 0.f;
    }
    return 0.f;
  }

  int AppearanceEngine::numSectors() const
  {
    switch (method_)
    {
    case AppearanceMethod::SC_PLUS_PLUS:
    case AppearanceMethod::SC_PLUS_PLUS_CUDA:
      return dlio::sc::SC_NS;
    case AppearanceMethod::STD:
    case AppearanceMethod::STD_CUDA:
      return 0;
    }
    return 0;
  }

} // namespace dlio
