#ifndef _MAP_POINT_HPP_
#define _MAP_POINT_HPP_

#include "feature.hpp"
#include "global.hpp"

namespace ssvo {

class Frame;

class KeyFrame;

class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:

    enum Type{
        BAD = -1,
        SEED = 0,
        STABLE = 1,
    };

    typedef std::shared_ptr<MapPoint> Ptr;

    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    typedef std::shared_ptr<Frame> FramePtr;

    Type type();

    void setBad();

    bool isBad();

    void resetType(Type type);

    KeyFramePtr getReferenceKeyFrame();

    bool fusion(const MapPoint::Ptr &mpt);

    void addObservation(const KeyFramePtr &kf, const Feature::Ptr &ft);

    int observations();

    std::map<KeyFramePtr, Feature::Ptr> getObservations();

    bool removeObservation(const KeyFramePtr &kf);

    Feature::Ptr findObservation(const KeyFramePtr kf);

    void updateViewAndDepth();

    int predictScale(const double dist, const int max_level);

    static int predictScale(const double dist_ref, const double dist_cur, const int level_ref, const int max_level);

    double getMinDistanceInvariance();

    double getMaxDistanceInvariance();

    bool getCloseViewObs(const FramePtr &frame, KeyFramePtr &keyframe, int &level);

    void increaseFound(int n=1);

    void increaseVisible(int n=1);

    uint64_t getFound();

    uint64_t getVisible();

    double getFoundRatio();

    inline void setPose(const double x, const double y, const double z)
    {
        pose_[0] = x;
        pose_[1] = y;
        pose_[2] = z;
    }

    inline void setPose(const Vector3d pose) { pose_ = pose; }

    inline Vector3d pose() { return pose_; }

    inline static Ptr create(const Vector3d &p)
    { return Ptr(new MapPoint(p)); }

private:

    MapPoint(const Vector3d &p);

    void updateRefKF();

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;

    static const double log_level_factor_;

    Vector3d optimal_pose_;
    double optimal_inv_z_;
    uint64_t last_structure_optimal_;

private:

    Vector3d pose_;

    std::unordered_map<KeyFramePtr, Feature::Ptr> obs_;

    Type type_;

    Vector3d obs_dir_; //!< mean viewing direction, from map point to keyframe
    double min_distance_;
    double max_distance_;

    KeyFramePtr refKF_;

    uint64_t found_cunter_;
    uint64_t visiable_cunter_;

    std::mutex mutex_obs_;
    std::mutex mutex_pose_;

};

typedef std::list<MapPoint::Ptr> MapPoints;

}


#endif