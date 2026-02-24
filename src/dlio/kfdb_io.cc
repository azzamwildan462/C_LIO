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

#include "dlio/kfdb_io.h"

#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace dlio::kfdb
{

  std::string getKfdbPath(const std::string &map_path, bool use_corrected)
  {
    if (map_path.empty())
      return "";

    size_t dot = map_path.rfind('.');
    std::string stem = (dot != std::string::npos && map_path.substr(dot) == ".pcd")
                           ? map_path.substr(0, dot)
                           : map_path;

    if (use_corrected)
    {
      std::string corrected = stem + "_corrected.kfdb";
      if (std::filesystem::exists(corrected))
        return corrected;
    }
    return stem + ".kfdb";
  }

  bool save(const std::string &path,
            const std::vector<dlio::sc::ScanContextEntry> &entries,
            float sc_max_range,
            const Eigen::Quaternionf &gravity_q)
  {
    if (path.empty() || entries.empty())
      return false;

    // Ensure parent directory exists
    std::filesystem::path filepath(path);
    if (filepath.has_parent_path())
      std::filesystem::create_directories(filepath.parent_path());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
      return false;

    // Header (v2: includes gravity quaternion)
    uint32_t magic = 0x4B464442; // "KFDB"
    uint32_t version = 2;
    uint32_t sc_nr = dlio::sc::SC_NR;
    uint32_t sc_ns = dlio::sc::SC_NS;
    uint32_t num_entries = static_cast<uint32_t>(entries.size());

    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char *>(&sc_nr), sizeof(sc_nr));
    ofs.write(reinterpret_cast<const char *>(&sc_ns), sizeof(sc_ns));
    ofs.write(reinterpret_cast<const char *>(&sc_max_range), sizeof(sc_max_range));
    ofs.write(reinterpret_cast<const char *>(&num_entries), sizeof(num_entries));

    float gq[4] = {gravity_q.w(), gravity_q.x(), gravity_q.y(), gravity_q.z()};
    ofs.write(reinterpret_cast<const char *>(gq), sizeof(gq));

    // Entries
    for (const auto &entry : entries)
    {
      ofs.write(reinterpret_cast<const char *>(entry.position.data()), 3 * sizeof(float));
      float qdata[4] = {entry.orientation.w(), entry.orientation.x(),
                        entry.orientation.y(), entry.orientation.z()};
      ofs.write(reinterpret_cast<const char *>(qdata), 4 * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(entry.descriptor.data()),
                dlio::sc::SC_NR * dlio::sc::SC_NS * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(entry.ring_key.data()),
                dlio::sc::SC_NR * sizeof(float));
    }

    ofs.close();
    return true;
  }

  bool saveCorrected(const std::string &map_path,
                     const std::vector<dlio::sc::ScanContextEntry> &entries,
                     const std::vector<geometry_msgs::msg::Pose> &corrected_poses,
                     float sc_max_range,
                     const Eigen::Quaternionf &gravity_q)
  {
    if (entries.empty() || corrected_poses.empty())
      return false;

    // Derive path: dlio_map.pcd → dlio_map_corrected.kfdb
    std::string kfdb_path;
    {
      size_t dot = map_path.rfind('.');
      if (dot != std::string::npos && map_path.substr(dot) == ".pcd")
        kfdb_path = map_path.substr(0, dot) + "_corrected.kfdb";
      else
        kfdb_path = map_path + "_corrected.kfdb";
    }

    uint32_t num_entries = static_cast<uint32_t>(
        std::min(entries.size(), corrected_poses.size()));

    std::filesystem::path filepath(kfdb_path);
    if (filepath.has_parent_path())
      std::filesystem::create_directories(filepath.parent_path());

    std::ofstream ofs(kfdb_path, std::ios::binary);
    if (!ofs.is_open())
      return false;

    uint32_t magic = 0x4B464442;
    uint32_t version = 2;
    uint32_t sc_nr = dlio::sc::SC_NR;
    uint32_t sc_ns = dlio::sc::SC_NS;

    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char *>(&sc_nr), sizeof(sc_nr));
    ofs.write(reinterpret_cast<const char *>(&sc_ns), sizeof(sc_ns));
    ofs.write(reinterpret_cast<const char *>(&sc_max_range), sizeof(sc_max_range));
    ofs.write(reinterpret_cast<const char *>(&num_entries), sizeof(num_entries));
    float gq[4] = {gravity_q.w(), gravity_q.x(), gravity_q.y(), gravity_q.z()};
    ofs.write(reinterpret_cast<const char *>(gq), sizeof(gq));

    for (uint32_t i = 0; i < num_entries; ++i)
    {
      const auto &entry = entries[i];
      const auto &pose = corrected_poses[i];

      float pos[3] = {static_cast<float>(pose.position.x),
                      static_cast<float>(pose.position.y),
                      static_cast<float>(pose.position.z)};
      ofs.write(reinterpret_cast<const char *>(pos), 3 * sizeof(float));

      float qdata[4] = {static_cast<float>(pose.orientation.w),
                        static_cast<float>(pose.orientation.x),
                        static_cast<float>(pose.orientation.y),
                        static_cast<float>(pose.orientation.z)};
      ofs.write(reinterpret_cast<const char *>(qdata), 4 * sizeof(float));

      ofs.write(reinterpret_cast<const char *>(entry.descriptor.data()),
                dlio::sc::SC_NR * dlio::sc::SC_NS * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(entry.ring_key.data()),
                dlio::sc::SC_NR * sizeof(float));
    }

    ofs.close();
    return true;
  }

  bool load(const std::string &path,
            std::vector<dlio::sc::ScanContextEntry> &database,
            float &sc_max_range,
            Eigen::Quaternionf &gravity_q)
  {
    if (path.empty())
      return false;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
      return false;

    uint32_t magic, version, sc_nr, sc_ns, num_entries;
    float file_max_range;

    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic != 0x4B464442)
      return false;

    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1 && version != 2)
      return false;

    ifs.read(reinterpret_cast<char *>(&sc_nr), sizeof(sc_nr));
    ifs.read(reinterpret_cast<char *>(&sc_ns), sizeof(sc_ns));
    ifs.read(reinterpret_cast<char *>(&file_max_range), sizeof(file_max_range));
    ifs.read(reinterpret_cast<char *>(&num_entries), sizeof(num_entries));

    // v2: gravity quaternion
    if (version >= 2)
    {
      float gq[4];
      ifs.read(reinterpret_cast<char *>(gq), sizeof(gq));
      gravity_q = Eigen::Quaternionf(gq[0], gq[1], gq[2], gq[3]);
    }

    if (sc_nr != dlio::sc::SC_NR || sc_ns != dlio::sc::SC_NS)
      return false;

    sc_max_range = file_max_range;

    database.clear();
    database.reserve(num_entries);

    for (uint32_t i = 0; i < num_entries; i++)
    {
      dlio::sc::ScanContextEntry entry;
      ifs.read(reinterpret_cast<char *>(entry.position.data()), 3 * sizeof(float));

      float qdata[4];
      ifs.read(reinterpret_cast<char *>(qdata), 4 * sizeof(float));
      entry.orientation = Eigen::Quaternionf(qdata[0], qdata[1], qdata[2], qdata[3]);

      entry.descriptor.resize(dlio::sc::SC_NR, dlio::sc::SC_NS);
      ifs.read(reinterpret_cast<char *>(entry.descriptor.data()),
               dlio::sc::SC_NR * dlio::sc::SC_NS * sizeof(float));

      entry.ring_key.resize(dlio::sc::SC_NR);
      ifs.read(reinterpret_cast<char *>(entry.ring_key.data()),
               dlio::sc::SC_NR * sizeof(float));

      // SC++: derive sector key from descriptor (backward compatible, no format change)
      entry.sector_key = dlio::sc::computeSectorKey(entry.descriptor);

      database.push_back(std::move(entry));
    }

    return true;
  }

} // namespace dlio::kfdb
