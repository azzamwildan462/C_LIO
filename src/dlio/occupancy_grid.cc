#include "dlio/occupancy_grid.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace dlio {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
OccupancyGridGenerator::OccupancyGridGenerator(const Params& params)
    : params_(params), initialized_(false)
{
  width_     = static_cast<int>(params_.grid_size_x / params_.resolution);
  height_    = static_cast<int>(params_.grid_size_y / params_.resolution);
  num_cells_ = width_ * height_;

  log_odds_.resize(num_cells_, 0.0f);
  last_update_time_.resize(num_cells_, 0.0);

  lo_occupied_ = std::log(static_cast<float>(params_.p_occupied / (1.0 - params_.p_occupied)));
  lo_free_     = std::log(static_cast<float>(params_.p_free     / (1.0 - params_.p_free)));
  lo_prior_    = std::log(static_cast<float>(params_.p_prior    / (1.0 - params_.p_prior)));

  num_angular_bins_   = 720;
  angular_resolution_ = 2.0f * static_cast<float>(M_PI) / static_cast<float>(num_angular_bins_);

  grid_center_  = Eigen::Vector2f(0.f, 0.f);
  grid_origin_x_ = 0.0;
  grid_origin_y_ = 0.0;
}

// ---------------------------------------------------------------------------
// Ground / obstacle classification
// ---------------------------------------------------------------------------
OccupancyGridGenerator::ClassifiedPoints
OccupancyGridGenerator::classifyPoints(
    pcl::PointCloud<PointType>::ConstPtr cloud,
    const Eigen::Vector3f& sensor_origin) const
{
  ClassifiedPoints result;
  result.obstacle_xy.reserve(cloud->size() / 4);
  result.all_xy.reserve(cloud->size());
  result.all_ranges.reserve(cloud->size());

  float sensor_z = sensor_origin.z() + static_cast<float>(params_.height_offset);

  for (const auto& pt : cloud->points) {
    float dx = pt.x - sensor_origin.x();
    float dy = pt.y - sensor_origin.y();
    float range_2d = std::sqrt(dx * dx + dy * dy);

    if (range_2d < 0.3f) continue;

    result.all_xy.emplace_back(pt.x, pt.y);
    result.all_ranges.push_back(range_2d);

    float height_rel = pt.z - sensor_z;
    if (height_rel > static_cast<float>(params_.obstacle_min_height) &&
        height_rel < static_cast<float>(params_.obstacle_max_height)) {
      result.obstacle_xy.emplace_back(pt.x, pt.y);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Polar coordinate binning
// ---------------------------------------------------------------------------
std::vector<OccupancyGridGenerator::PolarBin>
OccupancyGridGenerator::polarBin(
    const ClassifiedPoints& classified,
    const Eigen::Vector2f& sensor_2d) const
{
  std::vector<PolarBin> bins(num_angular_bins_);

  for (size_t i = 0; i < classified.all_xy.size(); i++) {
    float dx = classified.all_xy[i].x() - sensor_2d.x();
    float dy = classified.all_xy[i].y() - sensor_2d.y();
    float angle = std::atan2(dy, dx) + static_cast<float>(M_PI);  // [0, 2*PI]
    int bin_idx = static_cast<int>(angle / angular_resolution_);
    if (bin_idx >= num_angular_bins_) bin_idx = num_angular_bins_ - 1;
    if (bin_idx < 0) bin_idx = 0;

    float range = classified.all_ranges[i];
    if (range > bins[bin_idx].max_range)
      bins[bin_idx].max_range = range;
  }

  for (const auto& obs_pt : classified.obstacle_xy) {
    float dx = obs_pt.x() - sensor_2d.x();
    float dy = obs_pt.y() - sensor_2d.y();
    float angle = std::atan2(dy, dx) + static_cast<float>(M_PI);
    int bin_idx = static_cast<int>(angle / angular_resolution_);
    if (bin_idx >= num_angular_bins_) bin_idx = num_angular_bins_ - 1;
    if (bin_idx < 0) bin_idx = 0;

    float range = std::sqrt(dx * dx + dy * dy);
    bins[bin_idx].has_obstacle = true;
    if (range < bins[bin_idx].obstacle_range)
      bins[bin_idx].obstacle_range = range;
  }

  return bins;
}

// ---------------------------------------------------------------------------
// Adaptive ground segmentation + polar binning (slope-based)
// ---------------------------------------------------------------------------
std::vector<OccupancyGridGenerator::PolarBin>
OccupancyGridGenerator::classifyAndBinAdaptive(
    pcl::PointCloud<PointType>::ConstPtr cloud,
    const Eigen::Vector2f& sensor_2d,
    float sensor_z,
    float robot_yaw,
    float robot_pitch) const
{
  struct BinPoint {
    float x, y, z;
    float range_2d;
  };

  std::vector<std::vector<BinPoint>> bin_points(num_angular_bins_);

  for (const auto& pt : cloud->points) {
    float dx = pt.x - sensor_2d.x();
    float dy = pt.y - sensor_2d.y();
    float range_2d = std::sqrt(dx * dx + dy * dy);

    if (range_2d < 0.3f) continue;

    float angle = std::atan2(dy, dx) + static_cast<float>(M_PI);
    int bin_idx = static_cast<int>(angle / angular_resolution_);
    if (bin_idx >= num_angular_bins_) bin_idx = num_angular_bins_ - 1;
    if (bin_idx < 0) bin_idx = 0;

    bin_points[bin_idx].push_back({pt.x, pt.y, pt.z, range_2d});
  }

  float slope_thresh = static_cast<float>(params_.ground_slope_threshold * M_PI / 180.0);
  float obs_min = static_cast<float>(params_.obstacle_min_height);
  float obs_max = static_cast<float>(params_.obstacle_max_height);
  float initial_ground_z = sensor_z + static_cast<float>(params_.ground_threshold);

  std::vector<PolarBin> bins(num_angular_bins_);

  for (int i = 0; i < num_angular_bins_; i++) {
    auto& pts = bin_points[i];
    if (pts.empty()) continue;

    // Sort by range (closest first)
    std::sort(pts.begin(), pts.end(),
              [](const BinPoint& a, const BinPoint& b) {
                return a.range_2d < b.range_2d;
              });

    // Pitch compensation: expected Z gradient in this bin's direction
    // Forward (along robot heading) = full pitch, sideways = zero
    float bin_angle = static_cast<float>(i) * angular_resolution_
                    - static_cast<float>(M_PI);
    float dir_rel = bin_angle - robot_yaw;
    float expected_dz_per_dr = std::tan(robot_pitch * std::cos(dir_rel));

    // Adaptive ground estimation along this radial ray
    float ground_z       = initial_ground_z;
    float last_ground_z  = initial_ground_z;
    float last_ground_r  = 0.f;

    for (const auto& p : pts) {
      if (p.range_2d > bins[i].max_range)
        bins[i].max_range = p.range_2d;

      float dr = p.range_2d - last_ground_r;
      float dz = p.z - last_ground_z;

      // Subtract expected height change due to robot pitch
      float residual_dz = dz - expected_dz_per_dr * dr;
      float slope = (dr > 0.05f)
          ? std::atan2(std::abs(residual_dz), dr) : 0.f;

      // Expected ground height at this range (pitch-compensated)
      float expected_ground_z = ground_z
          + expected_dz_per_dr * (p.range_2d - last_ground_r);
      float height_above_ground = p.z - expected_ground_z;

      if (height_above_ground < obs_min) {
        // At or below expected ground level
        if (slope < slope_thresh) {
          ground_z      = p.z;
          last_ground_z = p.z;
          last_ground_r = p.range_2d;
        }
      } else if (height_above_ground < obs_max) {
        // Above expected ground — gentle residual = ground, steep = obstacle
        if (slope < slope_thresh) {
          ground_z      = p.z;
          last_ground_z = p.z;
          last_ground_r = p.range_2d;
        } else {
          bins[i].has_obstacle = true;
          if (p.range_2d < bins[i].obstacle_range)
            bins[i].obstacle_range = p.range_2d;
        }
      }
    }
  }

  return bins;
}

// ---------------------------------------------------------------------------
// Bresenham ray tracing
// ---------------------------------------------------------------------------
void OccupancyGridGenerator::rayTrace(int x0, int y0, int x1, int y1,
                                       bool mark_endpoint_occupied)
{
  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;

  int cx = x0, cy = y0;

  while (!(cx == x1 && cy == y1)) {
    if (inBounds(cx, cy)) {
      bayesUpdate(cellIndex(cx, cy), lo_free_);
    }

    int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; cx += sx; }
    if (e2 <  dx) { err += dx; cy += sy; }

    // Safety: break if ray went way too long (shouldn't happen normally)
    if (std::abs(cx - x0) > width_ || std::abs(cy - y0) > height_) break;
  }

  if (mark_endpoint_occupied && inBounds(x1, y1)) {
    bayesUpdate(cellIndex(x1, y1), lo_occupied_);

    int margin_cells = static_cast<int>(
        std::ceil(params_.obstacle_margin / params_.resolution));
    for (int mx = -margin_cells; mx <= margin_cells; mx++) {
      for (int my = -margin_cells; my <= margin_cells; my++) {
        if (mx == 0 && my == 0) continue;
        int nx = x1 + mx;
        int ny = y1 + my;
        if (inBounds(nx, ny)) {
          bayesUpdate(cellIndex(nx, ny), lo_occupied_ * 0.5f);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Binary Bayes filter update (log-odds)
// ---------------------------------------------------------------------------
void OccupancyGridGenerator::bayesUpdate(int cell_idx, float lo_measurement)
{
  float new_lo = log_odds_[cell_idx] + lo_measurement - lo_prior_;
  log_odds_[cell_idx] = std::clamp(
      new_lo,
      static_cast<float>(params_.lo_clamped_min),
      static_cast<float>(params_.lo_clamped_max));
}

// ---------------------------------------------------------------------------
// Grid shifting (rolling window)
// ---------------------------------------------------------------------------
void OccupancyGridGenerator::shiftGrid(const Eigen::Vector2f& new_center)
{
  double dx_world = new_center.x() - grid_center_.x();
  double dy_world = new_center.y() - grid_center_.y();

  int shift_x = static_cast<int>(std::round(dx_world / params_.resolution));
  int shift_y = static_cast<int>(std::round(dy_world / params_.resolution));

  if (shift_x == 0 && shift_y == 0) return;

  std::vector<float>  new_log_odds(num_cells_, 0.0f);
  std::vector<double> new_timestamps(num_cells_, 0.0);

  for (int gy = 0; gy < height_; gy++) {
    for (int gx = 0; gx < width_; gx++) {
      int src_x = gx + shift_x;
      int src_y = gy + shift_y;

      if (src_x >= 0 && src_x < width_ && src_y >= 0 && src_y < height_) {
        int src_idx = src_y * width_ + src_x;
        int dst_idx = gy * width_ + gx;
        new_log_odds[dst_idx]   = log_odds_[src_idx];
        new_timestamps[dst_idx] = last_update_time_[src_idx];
      }
    }
  }

  log_odds_         = std::move(new_log_odds);
  last_update_time_ = std::move(new_timestamps);

  grid_origin_x_ += shift_x * params_.resolution;
  grid_origin_y_ += shift_y * params_.resolution;
  grid_center_ = new_center;
}

// ---------------------------------------------------------------------------
// Main update
// ---------------------------------------------------------------------------
void OccupancyGridGenerator::update(
    pcl::PointCloud<PointType>::ConstPtr cloud,
    const Eigen::Vector3f& sensor_origin,
    double timestamp_sec,
    float robot_yaw,
    float robot_pitch)
{
  if (!cloud || cloud->empty()) return;

  std::lock_guard<std::mutex> lock(grid_mtx_);

  Eigen::Vector2f sensor_2d(sensor_origin.x(), sensor_origin.y());

  // Initialize or shift grid
  if (!initialized_) {
    grid_center_ = sensor_2d;
    grid_origin_x_ = static_cast<double>(sensor_2d.x()) - params_.grid_size_x / 2.0;
    grid_origin_y_ = static_cast<double>(sensor_2d.y()) - params_.grid_size_y / 2.0;
    initialized_ = true;
  } else {
    shiftGrid(sensor_2d);
  }

  // Time decay
  if (params_.decay_rate > 0.0) {
    for (int i = 0; i < num_cells_; i++) {
      if (last_update_time_[i] > 0.0 && log_odds_[i] != 0.0f) {
        double dt = timestamp_sec - last_update_time_[i];
        if (dt > 0.0) {
          float decay = 1.0f - std::exp(static_cast<float>(-params_.decay_rate * dt));
          log_odds_[i] *= (1.0f - decay);
        }
      }
    }
  }

  // 1+2. Classify + polar binning
  float sensor_z = sensor_origin.z() + static_cast<float>(params_.height_offset);
  std::vector<PolarBin> bins;

  if (params_.adaptive_ground) {
    bins = classifyAndBinAdaptive(cloud, sensor_2d, sensor_z, robot_yaw, robot_pitch);
  } else {
    auto classified = classifyPoints(cloud, sensor_origin);
    bins = polarBin(classified, sensor_2d);
  }

  // 3. Ray tracing
  int sensor_gx = worldToGridX(sensor_2d.x());
  int sensor_gy = worldToGridY(sensor_2d.y());

  // Precompute angular filter bounds (in radians)
  float angle_tol_rad = static_cast<float>(params_.scan_angle_tolerance * M_PI / 180.0);
  float angle_target_rad = static_cast<float>(params_.scan_angle_target * M_PI / 180.0);
  bool use_angular_filter = (params_.scan_angle_tolerance < 179.9);

  for (int i = 0; i < num_angular_bins_; i++) {
    if (bins[i].max_range < 0.5f) continue;

    // bin angle is in world frame (atan2-based, from -PI to PI)
    float angle = static_cast<float>(i) * angular_resolution_ - static_cast<float>(M_PI);

    // Angular filter: skip bins outside target ± tolerance relative to robot heading
    if (use_angular_filter) {
      // Compute angle relative to robot forward direction
      float rel_angle = angle - robot_yaw - angle_target_rad;
      // Normalize to [-PI, PI]
      while (rel_angle >  static_cast<float>(M_PI)) rel_angle -= 2.f * static_cast<float>(M_PI);
      while (rel_angle < -static_cast<float>(M_PI)) rel_angle += 2.f * static_cast<float>(M_PI);
      if (std::abs(rel_angle) > angle_tol_rad) continue;
    }

    if (bins[i].has_obstacle) {
      float obs_range = bins[i].obstacle_range;
      float end_x = sensor_2d.x() + obs_range * std::cos(angle);
      float end_y = sensor_2d.y() + obs_range * std::sin(angle);
      int end_gx = worldToGridX(end_x);
      int end_gy = worldToGridY(end_y);
      rayTrace(sensor_gx, sensor_gy, end_gx, end_gy, true);

      if (inBounds(end_gx, end_gy))
        last_update_time_[cellIndex(end_gx, end_gy)] = timestamp_sec;
    } else {
      float max_range = bins[i].max_range;
      float end_x = sensor_2d.x() + max_range * std::cos(angle);
      float end_y = sensor_2d.y() + max_range * std::sin(angle);
      int end_gx = worldToGridX(end_x);
      int end_gy = worldToGridY(end_y);
      rayTrace(sensor_gx, sensor_gy, end_gx, end_gy, false);
    }
  }
}

// ---------------------------------------------------------------------------
// Generate ROS message
// ---------------------------------------------------------------------------
nav_msgs::msg::OccupancyGrid
OccupancyGridGenerator::getOccupancyGrid(rclcpp::Time stamp) const
{
  std::lock_guard<std::mutex> lock(grid_mtx_);

  nav_msgs::msg::OccupancyGrid msg;
  msg.header.stamp    = stamp;
  msg.header.frame_id = params_.frame_id;

  msg.info.resolution = static_cast<float>(params_.resolution);
  msg.info.width  = static_cast<uint32_t>(width_);
  msg.info.height = static_cast<uint32_t>(height_);
  msg.info.origin.position.x = grid_origin_x_;
  msg.info.origin.position.y = grid_origin_y_;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.w = 1.0;

  msg.data.resize(num_cells_);
  for (int i = 0; i < num_cells_; i++) {
    if (std::abs(log_odds_[i]) < 0.01f) {
      msg.data[i] = -1;  // unknown
    } else {
      float prob = 1.0f / (1.0f + std::exp(-log_odds_[i]));
      msg.data[i] = static_cast<int8_t>(std::clamp(prob * 100.0f, 0.0f, 100.0f));
    }
  }

  return msg;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
void OccupancyGridGenerator::reset()
{
  std::lock_guard<std::mutex> lock(grid_mtx_);
  std::fill(log_odds_.begin(), log_odds_.end(), 0.0f);
  std::fill(last_update_time_.begin(), last_update_time_.end(), 0.0);
  initialized_ = false;
}

}  // namespace dlio
