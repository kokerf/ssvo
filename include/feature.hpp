#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

class MapPoint;
class KeyFrame;

class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Vector2d px_;
    Vector3d fn_;
    int level_;
    std::shared_ptr<MapPoint> mpt_;

    inline static Ptr create(const Vector2d &_px, const Vector3d &_fn, int _level, const std::shared_ptr<MapPoint> &_mpt)
    {return std::make_shared<Feature>(Feature(_px, _fn, _level, _mpt));}

private:
    Feature(const Vector2d &px, const Vector3d &fn, const int level, const std::shared_ptr<MapPoint> &mpt):
        px_(px), fn_(fn), level_(level), mpt_(mpt)
    {
        assert(fn[2] == 1);
        assert(mpt);
    }

};

typedef std::list<Feature::Ptr> Features;

}

#endif