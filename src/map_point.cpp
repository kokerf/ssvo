#include "map_point.hpp"
#include "keyframe.hpp"

namespace ssvo
{

unsigned long int MapPoint::next_id_ = 0;
const double MapPoint::log_level_factor_ = log(2.0f);

MapPoint::MapPoint(const Vector3d &p, const KeyFrame::Ptr &kf) :
        id_(next_id_++), pose_(p), n_obs_(0), min_distance_(0.0), max_distance_(0.0), refKF_(kf),
        found_cunter_(1), visiable_cunter_(1)
{
}

void MapPoint::addObservation(const KeyFrame::Ptr kf, const Feature::Ptr ft)
{
    obs_.insert(std::make_pair(kf, ft));
    n_obs_++;
}

std::map<KeyFrame::Ptr, Feature::Ptr> MapPoint::getObservations()
{
    return std::map<KeyFrame::Ptr, Feature::Ptr>(obs_.begin(), obs_.end());
}

Feature::Ptr MapPoint::findObservation(const KeyFrame::Ptr kf)
{
    auto it = obs_.find(kf);
    if(it == obs_.end())
        return nullptr;
    else
        return it->second;
}

void MapPoint::updateViewAndDepth()
{
    if(obs_.empty())
        return;

    Vector3d normal = Vector3d::Zero();
    int n = 0;
    for(std::pair<KeyFrame::Ptr, Feature::Ptr> obs : obs_)
    {
        Vector3d Ow = obs.first->pose().translation();
        Vector3d obs_dir((Ow - pose_).normalized());
        normal = normal + obs_dir;
        n++;
    }

    obs_dir_ = normal / n;

    Vector3d ref_obs_dir = refKF_->pose().translation() - pose_;
    const double dist = ref_obs_dir.norm();
    Feature::Ptr ft = findObservation(refKF_);
    const int level_scale = 1 << ft->level;
    const int max_scale = 1 << refKF_->nlevels_;

    // TODO whether change?
    max_distance_ = dist * level_scale; //! regard it is top level, we may obsevere the point if we go closer
    min_distance_ = max_distance_ / max_scale;
}

//double MapPoint::getMinDistanceInvariance()
//{
//    return 0.8f * min_distance_;
//}
//
//double MapPoint::getMaxDistanceInvariance()
//{
//    return 1.2f * max_distance_;
//}

int MapPoint::predictScale(const double dist, const int max_level) const
{
    double ratio = max_distance_ / dist;

    int scale = round(log(ratio) / log_level_factor_);
    if(scale < 0)
        scale = 0;
    else if(scale > max_level)
        scale = max_level;

    return scale;
}

int MapPoint::predictScale(const double dist_ref, const double dist_cur, const int level_ref, const int max_level)
{
    double ratio = dist_ref * (1 << level_ref) / dist_cur;

    int scale = round(log(ratio) / log_level_factor_);
    if(scale < 0)
        scale = 0;
    else if(scale > max_level)
        scale = max_level;

    return scale;
}

void MapPoint::increaseFound(int n)
{
    found_cunter_ += n;
}

void MapPoint::increaseVisible(int n)
{
    visiable_cunter_ += n;
}

double MapPoint::getFoundRatio()
{
    return static_cast<double>(found_cunter_)/visiable_cunter_;
}

bool MapPoint::getCloseViewObs(const Frame::Ptr &frame, KeyFrame::Ptr &keyframe, int &level)
{
    LOG_ASSERT(!obs_.empty()) << "Map point is invalid!";

    //! 1. scale invariance region check
    Vector3d frame_obs_dir = frame->pose().translation() - pose_;
    const double dist = frame_obs_dir.norm();
    if(dist < 0.8f*min_distance_ || dist > 1.2f*max_distance_)
        return false;

    //! 2. observation view check
    frame_obs_dir.normalize();
    if(frame_obs_dir.dot(obs_dir_) < 0.5)
        return false;

    //! 3. find close view keyframe
    Vector3d frame_dir(frame->ray().normalized());

    double max_cos_angle = 0.0;
    for(std::pair<KeyFrame::Ptr, Feature::Ptr> item : obs_)
    {
        Vector3d kf_dir(item.first->ray().normalized());
        double view_cos_angle = kf_dir.dot(frame_dir);

        //! find min angle
        if(view_cos_angle < max_cos_angle)
            continue;

        max_cos_angle = view_cos_angle;
        keyframe = item.first;
    }

    if(max_cos_angle < 0.5f)
        return false;

    level = predictScale(dist, frame->nlevels_-1);

    return true;
}

}
