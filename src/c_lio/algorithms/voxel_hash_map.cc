#include "c_lio/algorithms/voxel_hash_map.h"
#include <algorithm>

namespace c_lio
{

  VoxelHashMap::VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
      : voxel_size_(voxel_size),
        max_distance_(max_distance),
        max_points_per_voxel_(max_points_per_voxel)
  {
    map_resolution_ = std::sqrt(voxel_size_ * voxel_size_ / max_points_per_voxel_);
  }

  void VoxelHashMap::update(pcl::PointCloud<PointType>::ConstPtr cloud,
                            const Eigen::Vector3f &origin)
  {
    addPoints(cloud);
    removeFarPoints(origin);
  }

  void VoxelHashMap::addPoints(pcl::PointCloud<PointType>::ConstPtr cloud)
  {
    for (const auto &pt : cloud->points)
    {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      const auto voxel = pointToVoxel(pt);
      auto it = map_.find(voxel);

      if (it != map_.end())
      {
        auto &voxel_points = it->second;

        // Skip if voxel full
        if (static_cast<int>(voxel_points.size()) >= max_points_per_voxel_)
          continue;

        // Skip if too close to existing point in voxel
        bool too_close = false;
        for (const auto &vp : voxel_points)
        {
          float dx = pt.x - vp.x;
          float dy = pt.y - vp.y;
          float dz = pt.z - vp.z;
          if (std::sqrt(dx * dx + dy * dy + dz * dz) < map_resolution_)
          {
            too_close = true;
            break;
          }
        }
        if (too_close)
          continue;

        voxel_points.push_back(pt);
      }
      else
      {
        std::vector<PointType> voxel_points;
        voxel_points.reserve(max_points_per_voxel_);
        voxel_points.push_back(pt);
        map_.emplace(voxel, std::move(voxel_points));
      }
    }
  }

  void VoxelHashMap::removeFarPoints(const Eigen::Vector3f &origin)
  {
    double max_dist2 = max_distance_ * max_distance_;

    for (auto it = map_.begin(); it != map_.end();)
    {
      const auto &pt = it->second.front();
      float dx = pt.x - origin[0];
      float dy = pt.y - origin[1];
      float dz = pt.z - origin[2];
      if (static_cast<double>(dx * dx + dy * dy + dz * dz) >= max_dist2)
      {
        it = map_.erase(it);
      }
      else
      {
        ++it;
      }
    }
  }

  pcl::PointCloud<PointType>::Ptr VoxelHashMap::getCloud() const
  {
    auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
    cloud->points.reserve(map_.size() * max_points_per_voxel_);

    for (const auto &[voxel, points] : map_)
    {
      cloud->points.insert(cloud->points.end(), points.begin(), points.end());
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
  }

  void VoxelHashMap::clear()
  {
    map_.clear();
  }

  size_t VoxelHashMap::size() const
  {
    size_t n = 0;
    for (const auto &[v, pts] : map_)
      n += pts.size();
    return n;
  }

} // namespace c_lio
