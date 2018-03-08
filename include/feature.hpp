#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

class MapPoint;
class Seed;

class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Vector2d px_;
    Vector3d fn_;
    int level_;
    std::shared_ptr<MapPoint> mpt_;
    std::shared_ptr<Seed> seed_;

    inline static Ptr create(const Vector2d &px, const Vector3d &fn, int level, const std::shared_ptr<MapPoint> &mpt)
    {return std::make_shared<Feature>(Feature(px, fn, level, mpt));}

    inline static Ptr create(const Vector2d &px, const std::shared_ptr<MapPoint> &mpt)
    {return std::make_shared<Feature>(Feature(px, mpt));}

    inline static Ptr create(const Vector2d &px, int level, const std::shared_ptr<Seed> &seed)
    {return std::make_shared<Feature>(Feature(px, level, seed));}

private:

    Feature(const Vector2d &px, const Vector3d &fn, const int level, const std::shared_ptr<MapPoint> &mpt):
        px_(px), fn_(fn), level_(level), mpt_(mpt), seed_(nullptr)
    {
        assert(fn[2] == 1);
        assert(mpt);
    }

    Feature(const Vector2d &px, const std::shared_ptr<MapPoint> &mpt):
        px_(px), fn_(0,0,0), level_(0), mpt_(mpt), seed_(nullptr)
    {
        assert(mpt);
    }

    Feature(const Vector2d &px, int level, const std::shared_ptr<Seed> &seed):
        px_(px), fn_(0,0,0), level_(level), mpt_(nullptr), seed_(seed)
    {
    }

};

typedef std::list<Feature::Ptr> Features;

}

#endif