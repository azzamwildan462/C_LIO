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

#ifndef DLIO_SCAN_CONTEXT_H
#define DLIO_SCAN_CONTEXT_H

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <limits>
#include <utility>

#include "dlio/dlio.h"

namespace dlio::sc
{

  constexpr int SC_NR = 40;  // number of rings  (was 20)
  constexpr int SC_NS = 120; // number of sectors (was 60)

  using ScanContextDescriptor = Eigen::MatrixXf; // NR x NS
  using RingKey = Eigen::VectorXf;               // NR

  struct ScanContextEntry
  {
    ScanContextDescriptor descriptor;
    RingKey ring_key;
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
  };

  inline ScanContextDescriptor computeScanContext(
      pcl::PointCloud<PointType>::ConstPtr cloud, float max_range)
  {
    ScanContextDescriptor desc = ScanContextDescriptor::Zero(SC_NR, SC_NS);
    float ring_step = max_range / static_cast<float>(SC_NR);
    float sector_step = 2.f * M_PI / static_cast<float>(SC_NS);

    for (const auto &pt : cloud->points)
    {
      float range = std::sqrt(pt.x * pt.x + pt.y * pt.y);
      if (range < 1e-3f || range > max_range)
        continue;

      float angle = std::atan2(pt.y, pt.x) + M_PI; // [0, 2*PI]
      int ri = std::min(static_cast<int>(range / ring_step), SC_NR - 1);
      int si = std::min(static_cast<int>(angle / sector_step), SC_NS - 1);

      if (pt.z > desc(ri, si))
      {
        desc(ri, si) = pt.z;
      }
    }
    return desc;
  }

  inline RingKey computeRingKey(const ScanContextDescriptor &desc)
  {
    RingKey key(SC_NR);
    for (int r = 0; r < SC_NR; r++)
    {
      float sum = 0.f;
      int count = 0;
      for (int s = 0; s < SC_NS; s++)
      {
        if (desc(r, s) != 0.f)
        {
          sum += desc(r, s);
          count++;
        }
      }
      key(r) = (count > 0) ? sum / static_cast<float>(count) : 0.f;
    }
    return key;
  }

  inline std::pair<float, int> computeScanContextDistance(
      const ScanContextDescriptor &a, const ScanContextDescriptor &b)
  {
    // Try all sector shifts and find minimum column-wise cosine distance
    float best_dist = std::numeric_limits<float>::max();
    int best_shift = 0;

    for (int shift = 0; shift < SC_NS; shift++)
    {
      float total_dist = 0.f;
      int valid_cols = 0;

      for (int s = 0; s < SC_NS; s++)
      {
        int s_shifted = (s + shift) % SC_NS;
        Eigen::VectorXf col_a = a.col(s);
        Eigen::VectorXf col_b = b.col(s_shifted);

        float norm_a = col_a.norm();
        float norm_b = col_b.norm();

        if (norm_a < 1e-6f || norm_b < 1e-6f)
          continue;

        float cosine_sim = col_a.dot(col_b) / (norm_a * norm_b);
        cosine_sim = std::max(-1.f, std::min(1.f, cosine_sim));
        total_dist += 1.f - cosine_sim;
        valid_cols++;
      }

      if (valid_cols > 0)
      {
        float avg_dist = total_dist / static_cast<float>(valid_cols);
        if (avg_dist < best_dist)
        {
          best_dist = avg_dist;
          best_shift = shift;
        }
      }
    }
    return {best_dist, best_shift};
  }

  // Prepare a gravity-aligned scan for SC descriptor computation.
  // Rotates scan from body/lidar frame to gravity-aligned frame and normalizes Z.
  inline pcl::PointCloud<PointType>::Ptr prepareGravityAlignedScan(
      pcl::PointCloud<PointType>::ConstPtr scan,
      const Eigen::Quaternionf &gravity_q,
      const Eigen::Matrix3f &R_body_to_lidar)
  {
    Eigen::Matrix3f R_gravity = gravity_q.toRotationMatrix();
    Eigen::Matrix3f R_to_world = R_gravity * R_body_to_lidar;

    auto sc_scan = std::make_shared<pcl::PointCloud<PointType>>();
    sc_scan->points.resize(scan->points.size());
    for (size_t i = 0; i < scan->points.size(); i++)
    {
      Eigen::Vector3f p(scan->points[i].x, scan->points[i].y, scan->points[i].z);
      p = R_to_world * p;
      sc_scan->points[i].x = p[0];
      sc_scan->points[i].y = p[1];
      sc_scan->points[i].z = p[2];
    }

    // Normalize Z relative to ground (min Z)
    float z_min = std::numeric_limits<float>::max();
    for (const auto &pt : sc_scan->points)
      if (pt.z < z_min)
        z_min = pt.z;
    for (auto &pt : sc_scan->points)
      pt.z -= z_min;

    return sc_scan;
  }

} // namespace dlio::sc

#endif // DLIO_SCAN_CONTEXT_H
