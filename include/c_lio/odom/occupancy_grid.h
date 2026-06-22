#ifndef C_LIO_OCCUPANCY_GRID_H_
#define C_LIO_OCCUPANCY_GRID_H_

#include <vector>
#include <cstdint>
#include <mutex>
#include <cmath>
#include <string>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/time.hpp>

#include "c_lio/c_lio.h"

namespace c_lio {

class OccupancyGridGenerator {
public:
  struct Params {
    double grid_size_x = 100.0;
    double grid_size_y = 100.0;
    double resolution  = 0.2;

    double ground_threshold    = -0.3;
    double obstacle_min_height = 0.1;
    double obstacle_max_height = 3.0;

    double p_occupied = 0.7;
    double p_free     = 0.3;
    double p_prior    = 0.5;

    double lo_clamped_min = -4.0;
    double lo_clamped_max =  4.0;

    double decay_rate      = 0.0;
    double obstacle_margin = 0.3;
    double height_offset   = 0.0;  // offset from lidar height for classification reference

    // Adaptive ground segmentation (slope-based)
    bool   adaptive_ground         = false;
    double ground_slope_threshold  = 10.0;  // degrees — max slope to consider ground

    // Angular filter (degrees, relative to robot forward)
    // target=0 means forward, tolerance=180 means full 360° (no filter)
    double scan_angle_target    = 0.0;
    double scan_angle_tolerance = 180.0;

    std::string frame_id = "odom";
  };

  explicit OccupancyGridGenerator(const Params& params);

  void update(pcl::PointCloud<PointType>::ConstPtr cloud,
              const Eigen::Vector3f& sensor_origin,
              double timestamp_sec,
              float robot_yaw = 0.f,
              float robot_pitch = 0.f);

  nav_msgs::msg::OccupancyGrid getOccupancyGrid(rclcpp::Time stamp) const;

  void reset();

private:
  struct ClassifiedPoints {
    std::vector<Eigen::Vector2f> obstacle_xy;
    std::vector<Eigen::Vector2f> all_xy;
    std::vector<float> all_ranges;
  };

  struct PolarBin {
    float max_range       = 0.f;
    bool  has_obstacle    = false;
    float obstacle_range  = std::numeric_limits<float>::max();
  };

  ClassifiedPoints classifyPoints(pcl::PointCloud<PointType>::ConstPtr cloud,
                                  const Eigen::Vector3f& sensor_origin) const;

  std::vector<PolarBin> polarBin(const ClassifiedPoints& classified,
                                 const Eigen::Vector2f& sensor_2d) const;

  std::vector<PolarBin> classifyAndBinAdaptive(
      pcl::PointCloud<PointType>::ConstPtr cloud,
      const Eigen::Vector2f& sensor_2d,
      float sensor_z,
      float robot_yaw,
      float robot_pitch) const;

  void rayTrace(int x0, int y0, int x1, int y1, bool mark_endpoint_occupied);

  void bayesUpdate(int cell_idx, float lo_measurement);

  void shiftGrid(const Eigen::Vector2f& new_center);

  inline int worldToGridX(float wx) const {
    return static_cast<int>(std::floor((wx - grid_origin_x_) / params_.resolution));
  }
  inline int worldToGridY(float wy) const {
    return static_cast<int>(std::floor((wy - grid_origin_y_) / params_.resolution));
  }
  inline bool inBounds(int gx, int gy) const {
    return gx >= 0 && gx < width_ && gy >= 0 && gy < height_;
  }
  inline int cellIndex(int gx, int gy) const {
    return gy * width_ + gx;
  }

  Params params_;

  int width_;
  int height_;
  int num_cells_;

  double grid_origin_x_;
  double grid_origin_y_;
  Eigen::Vector2f grid_center_;

  std::vector<float>  log_odds_;
  std::vector<double> last_update_time_;

  float lo_occupied_;
  float lo_free_;
  float lo_prior_;

  int   num_angular_bins_;
  float angular_resolution_;

  bool initialized_;

  mutable std::mutex grid_mtx_;
};

}  // namespace c_lio

#endif  // C_LIO_OCCUPANCY_GRID_H_
