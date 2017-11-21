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

    MapPoint(const Vector3d p);

    MapPoint(const Vector3d p, const std::shared_ptr<KeyFrame> kf, const Feature::Ptr ft);

    MapPoint &operator=(const MapPoint&) = delete; //! copy denied

    void addObservation(const std::shared_ptr<KeyFrame> kf, const Feature::Ptr ft);

    Feature::Ptr findObservation(const std::shared_ptr<KeyFrame> kf);

    inline void setPose(const double x, const double y, const double z)
    {
        pose_[0] = x;
        pose_[1] = y;
        pose_[2] = z;
    }

    inline void setPose(const Vector3d pose) {pose_ = pose;}

    inline Vector3d pose() {return pose_;}

    inline static MapPoint::Ptr create(const Vector3d p) {return MapPoint::Ptr(new MapPoint(p));}

    inline static MapPoint::Ptr create(const Vector3d p, const std::shared_ptr<KeyFrame> kf, const Feature::Ptr ft) {return MapPoint::Ptr(new MapPoint(p, kf, ft));}

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;

private:
    Vector3d pose_;
    std::unordered_map<std::shared_ptr<KeyFrame>, Feature::Ptr> obs_;
    int n_obs_;

};

}


#endif