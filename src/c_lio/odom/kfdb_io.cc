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

#include "c_lio/odom/kfdb_io.h"
#include "c_lio/odom/scan_context.h"

#include <fstream>
#include <filesystem>
#include <system_error>
#include <cmath>
#include <algorithm>

namespace c_lio::kfdb
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
            const std::vector<c_lio::AppearanceEntry> &entries,
            float sc_max_range,
            const Eigen::Quaternionf &gravity_q)
  {
    if (path.empty() || entries.empty())
      return false;

    // Ensure parent directory exists
    std::filesystem::path filepath(path);
    if (filepath.has_parent_path())
      std::filesystem::create_directories(filepath.parent_path());

    // Atomic snapshot write: serialize to a temp file, then rename over the
    // real path. A crash/kill mid-write leaves the previous valid file intact
    // (rename is atomic on the same filesystem) — never a half-written .kfdb.
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs.is_open())
      return false;

    // Header (v4: v3 fields + per-entry gps_horizontal_accuracy + gps_status)
    uint32_t magic = 0x4B464442; // "KFDB"
    uint32_t version = 4;
    uint32_t sc_nr = c_lio::sc::SC_NR;
    uint32_t sc_ns = c_lio::sc::SC_NS;
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
      ofs.write(reinterpret_cast<const char *>(entry.descriptor.sc_descriptor.data()),
                c_lio::sc::SC_NR * c_lio::sc::SC_NS * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(entry.descriptor.ring_key.data()),
                c_lio::sc::SC_NR * sizeof(float));

      // v3: GPS fields
      ofs.write(reinterpret_cast<const char *>(&entry.gps_latitude), sizeof(double));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_longitude), sizeof(double));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_altitude), sizeof(double));
      uint8_t gv = entry.gps_valid ? 1 : 0;
      ofs.write(reinterpret_cast<const char *>(&gv), sizeof(uint8_t));

      // v4: GPS horizontal accuracy + status
      ofs.write(reinterpret_cast<const char *>(&entry.gps_horizontal_accuracy), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_status), sizeof(int8_t));
    }

    ofs.close();
    if (!ofs) // write/flush error — discard temp, keep previous file
    {
      std::error_code ec;
      std::filesystem::remove(tmp_path, ec);
      return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec)
    {
      std::filesystem::remove(tmp_path, ec);
      return false;
    }
    return true;
  }

  bool saveCorrected(const std::string &map_path,
                     const std::vector<c_lio::AppearanceEntry> &entries,
                     const std::vector<geometry_msgs::msg::Pose> &corrected_poses,
                     float sc_max_range,
                     const Eigen::Quaternionf &gravity_q)
  {
    if (entries.empty() || corrected_poses.empty())
      return false;

    // Derive path: c_lio_map.pcd → c_lio_map_corrected.kfdb
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

    // Atomic snapshot write (temp + rename) — see save() above.
    const std::string tmp_path = kfdb_path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs.is_open())
      return false;

    uint32_t magic = 0x4B464442;
    uint32_t version = 4;
    uint32_t sc_nr = c_lio::sc::SC_NR;
    uint32_t sc_ns = c_lio::sc::SC_NS;

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

      ofs.write(reinterpret_cast<const char *>(entry.descriptor.sc_descriptor.data()),
                c_lio::sc::SC_NR * c_lio::sc::SC_NS * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(entry.descriptor.ring_key.data()),
                c_lio::sc::SC_NR * sizeof(float));

      // v3: GPS fields
      ofs.write(reinterpret_cast<const char *>(&entry.gps_latitude), sizeof(double));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_longitude), sizeof(double));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_altitude), sizeof(double));
      uint8_t gv = entry.gps_valid ? 1 : 0;
      ofs.write(reinterpret_cast<const char *>(&gv), sizeof(uint8_t));

      // v4: GPS horizontal accuracy + status
      ofs.write(reinterpret_cast<const char *>(&entry.gps_horizontal_accuracy), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&entry.gps_status), sizeof(int8_t));
    }

    ofs.close();
    if (!ofs) // write/flush error — discard temp, keep previous file
    {
      std::error_code ec;
      std::filesystem::remove(tmp_path, ec);
      return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, kfdb_path, ec);
    if (ec)
    {
      std::filesystem::remove(tmp_path, ec);
      return false;
    }
    return true;
  }

  bool load(const std::string &path,
            std::vector<c_lio::AppearanceEntry> &database,
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
    if (version != 1 && version != 2 && version != 3 && version != 4)
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

    if (sc_nr != c_lio::sc::SC_NR || sc_ns != c_lio::sc::SC_NS)
      return false;

    sc_max_range = file_max_range;

    database.clear();
    database.reserve(num_entries);

    for (uint32_t i = 0; i < num_entries; i++)
    {
      c_lio::AppearanceEntry entry;
      ifs.read(reinterpret_cast<char *>(entry.position.data()), 3 * sizeof(float));

      float qdata[4];
      ifs.read(reinterpret_cast<char *>(qdata), 4 * sizeof(float));
      entry.orientation = Eigen::Quaternionf(qdata[0], qdata[1], qdata[2], qdata[3]);

      entry.descriptor.sc_descriptor.resize(c_lio::sc::SC_NR, c_lio::sc::SC_NS);
      ifs.read(reinterpret_cast<char *>(entry.descriptor.sc_descriptor.data()),
               c_lio::sc::SC_NR * c_lio::sc::SC_NS * sizeof(float));

      entry.descriptor.ring_key.resize(c_lio::sc::SC_NR);
      ifs.read(reinterpret_cast<char *>(entry.descriptor.ring_key.data()),
               c_lio::sc::SC_NR * sizeof(float));

      // SC++: derive sector key from descriptor (backward compatible, no format change)
      entry.descriptor.sector_key = c_lio::sc::computeSectorKey(entry.descriptor.sc_descriptor);

      // v3: GPS fields
      if (version >= 3)
      {
        ifs.read(reinterpret_cast<char *>(&entry.gps_latitude), sizeof(double));
        ifs.read(reinterpret_cast<char *>(&entry.gps_longitude), sizeof(double));
        ifs.read(reinterpret_cast<char *>(&entry.gps_altitude), sizeof(double));
        uint8_t gv = 0;
        ifs.read(reinterpret_cast<char *>(&gv), sizeof(uint8_t));
        entry.gps_valid = (gv != 0);
      }
      else
      {
        entry.gps_valid = false;
      }

      // v4: GPS horizontal accuracy + status
      if (version >= 4)
      {
        ifs.read(reinterpret_cast<char *>(&entry.gps_horizontal_accuracy), sizeof(float));
        ifs.read(reinterpret_cast<char *>(&entry.gps_status), sizeof(int8_t));
      }
      else
      {
        entry.gps_horizontal_accuracy = 0.f;
        entry.gps_status = entry.gps_valid ? static_cast<int8_t>(0) : static_cast<int8_t>(-1);
      }

      database.push_back(std::move(entry));
    }

    return true;
  }

} // namespace c_lio::kfdb
