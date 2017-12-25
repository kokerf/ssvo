#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

class MapPoint;

class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Vector2d px;
    Vector3d fn;
    uint8_t level;
    std::shared_ptr<MapPoint> mpt;

    inline static Feature::Ptr create(const Vector2d _px, const Vector3d _fn, const int _level, const std::shared_ptr<MapPoint> _mpt)
    {return std::make_shared<Feature>(Feature(_px, _fn, _level, _mpt));}

private:
    Feature(const Vector2d _px, const Vector3d _fn, const int _level, const std::shared_ptr<MapPoint> _mpt):
        px(_px), fn(_fn), level(_level), mpt(_mpt) {}
};

typedef std::list<Feature::Ptr> Features;

}

#endif