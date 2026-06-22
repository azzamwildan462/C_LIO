/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 *          Azzam Wildan M (SCLC extensions)               *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#ifndef C_LIO_C_LIO_H
#define C_LIO_C_LIO_H

// SYSTEM
#include <atomic>

#ifdef HAS_CPUID
#include <cpuid.h>
#endif

#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/times.h>
#include <thread>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

// BOOST
#include <boost/format.hpp>

// PCL
#define PCL_NO_PRECOMPILE

// C_LIO
#include "c_lio/algorithms/nano_gicp/nano_gicp.h"

namespace c_lio
{
  enum class SensorType
  {
    OUSTER,
    VELODYNE,
    HESAI,
    LIVOX,
    ROBOSENSE,
    UNKNOWN
  };

  class OdomNode;
  class GraphSlamNode;
  class LioSamMapOptimizationNode;

  struct Point
  {
    Point() : data{0.f, 0.f, 0.f, 1.f} {}

    PCL_ADD_POINT4D;
    float intensity; // intensity
    union
    {
      std::uint32_t t;  // (Ouster) time since beginning of scan in nanoseconds
      float time;       // (Velodyne) time since beginning of scan in seconds
      double timestamp; // (Hesai) absolute timestamp in seconds
                        // (Livox) absolute timestamp in (seconds * 10e9)
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT(c_lio::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint32_t, t, t)(float, time, time)(double, timestamp, timestamp))

typedef c_lio::Point PointType;

#endif // C_LIO_C_LIO_H
