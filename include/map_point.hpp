#ifndef _MAP_POINT_HPP_
#define _MAP_POINT_HPP_

#include "feature.hpp"
#include "global.hpp"

namespace ssvo {

class KeyFrame;

class MapPoint
{
public:

    typedef std::shared_ptr<MapPoint> Ptr;

    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    void addObservation(const KeyFramePtr kf, const Feature::Ptr ft);

    std::map<KeyFramePtr, Feature::Ptr> getObservations();

    Feature::Ptr findObservation(const KeyFramePtr kf);

    inline void setPose(const double x, const double y, const double z)
    {
        pose_[0] = x;
        pose_[1] = y;
        pose_[2] = z;
    }

    inline void setPose(const Vector3d pose) {pose_ = pose;}

    inline Vector3d pose() {return pose_;}

    inline static MapPoint::Ptr create(const Vector3d p)
    {return std::make_shared<MapPoint>(MapPoint(p));}

    inline static MapPoint::Ptr create(const Vector3d p, const KeyFramePtr kf, const Feature::Ptr ft)
    {return std::make_shared<MapPoint>(MapPoint(p, kf, ft));}

private:
    MapPoint(const Vector3d p);

    MapPoint(const Vector3d p, const KeyFramePtr kf, const Feature::Ptr ft);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;

    Vector3d optimal_pose_;

private:
    Vector3d pose_;
    std::unordered_map<KeyFramePtr, Feature::Ptr> obs_;
    int n_obs_;

};

typedef std::vector<MapPoint::Ptr> MapPoints;

}


#endif