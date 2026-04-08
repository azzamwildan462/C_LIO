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

#ifndef DLIO_KFDB_IO_H
#define DLIO_KFDB_IO_H

#include "dlio/scan_context.h"
#include <geometry_msgs/msg/pose.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace dlio::kfdb
{

    // Derive .kfdb path from .pcd map path.
    // If use_corrected is true, returns corrected variant if it exists on disk.
    std::string getKfdbPath(const std::string &map_path, bool use_corrected);

    // Save KFDB entries to binary file.
    // Returns true on success.
    bool save(const std::string &path,
              const std::vector<dlio::sc::ScanContextEntry> &entries,
              float sc_max_range,
              const Eigen::Quaternionf &gravity_q);

    // Save KFDB with corrected poses from lio_sam_opt.
    // Derives path as <stem>_corrected.kfdb from map_path.
    // Returns true on success.
    bool saveCorrected(const std::string &map_path,
                       const std::vector<dlio::sc::ScanContextEntry> &entries,
                       const std::vector<geometry_msgs::msg::Pose> &corrected_poses,
                       float sc_max_range,
                       const Eigen::Quaternionf &gravity_q);

    // Load KFDB from binary file into database vector.
    // sc_max_range and gravity_q are output parameters read from the file header.
    // Returns true on success.
    bool load(const std::string &path,
              std::vector<dlio::sc::ScanContextEntry> &database,
              float &sc_max_range,
              Eigen::Quaternionf &gravity_q);

} // namespace dlio::kfdb

#endif // DLIO_KFDB_IO_H
