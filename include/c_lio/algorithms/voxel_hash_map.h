#ifndef C_LIO_VOXEL_HASH_MAP_H
#define C_LIO_VOXEL_HASH_MAP_H

#include "c_lio/c_lio.h"
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace c_lio
{

  class VoxelHashMap
  {
  public:
    VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel);

    // Add points from a new scan and prune far voxels
    void update(pcl::PointCloud<PointType>::ConstPtr cloud, const Eigen::Vector3f &origin);

    // Extract accumulated cloud
    pcl::PointCloud<PointType>::Ptr getCloud() const;

    void clear();
    bool empty() const { return map_.empty(); }
    size_t size() const; // total point count

  private:
    using Voxel = Eigen::Vector3i;

    struct VoxelHash
    {
      size_t operator()(const Voxel &v) const
      {
        const auto *d = reinterpret_cast<const uint32_t *>(v.data());
        return static_cast<size_t>(d[0] * 73856093u ^ d[1] * 19349669u ^ d[2] * 83492791u);
      }
    };

    struct VoxelEqual
    {
      bool operator()(const Voxel &a, const Voxel &b) const
      {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
      }
    };

    Voxel pointToVoxel(const PointType &pt) const
    {
      return Voxel(static_cast<int>(std::floor(pt.x / voxel_size_)),
                   static_cast<int>(std::floor(pt.y / voxel_size_)),
                   static_cast<int>(std::floor(pt.z / voxel_size_)));
    }

    void addPoints(pcl::PointCloud<PointType>::ConstPtr cloud);
    void removeFarPoints(const Eigen::Vector3f &origin);

    std::unordered_map<Voxel, std::vector<PointType>, VoxelHash, VoxelEqual> map_;
    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    double map_resolution_; // min distance between points in same voxel
  };

} // namespace c_lio

#endif // C_LIO_VOXEL_HASH_MAP_H
