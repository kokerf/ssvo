#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

class MapPoint;

struct Feature
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Vector2d px;
    Vector3d ft;
    uint8_t level;
    std::shared_ptr<MapPoint> mpt;

    Feature(const Vector2d _px, const Vector3d _ft, const int _level = 0, const std::shared_ptr<MapPoint> _mpt= nullptr):
            px(_px), ft(_ft), level(_level), mpt(_mpt)
    {}

    inline static Feature::Ptr create(const Vector2d _px, const Vector3d _fx, const int _level = 0, const std::shared_ptr<MapPoint> _mpt= nullptr) {return Feature::Ptr(new Feature(_px, _fx, _level, _mpt));}

};

}

#endif