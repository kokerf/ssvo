#ifndef _GLOBAL_HPP_
#define _GLOBAL_HPP_

#include <cstdlib>
#include <stdint.h>
#include <assert.h>
#include <cmath>

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <random>
#include <list>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <glog/logging.h>

#include<Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3d)

using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

#ifndef MIN
    #define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
    #define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef std::vector<cv::Mat> ImgPyr;

namespace ssvo{

static std::mt19937_64 rd;
static std::uniform_real_distribution<double> distribution(0.0, std::nextafter(1, std::numeric_limits<double>::max()));

inline double Rand(double min, double max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}

inline int Rand(int min, int max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}

class noncopyable
{
protected:
    noncopyable() = default;
    ~noncopyable() = default;

    noncopyable(const noncopyable&) = delete;
    noncopyable &operator=(const noncopyable&) = delete;
};

}

#endif