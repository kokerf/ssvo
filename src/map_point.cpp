#include "map_point.hpp"
#include "keyframe.hpp"

namespace ssvo
{

unsigned long int MapPoint::next_id_ = 0;

MapPoint::MapPoint(const Vector3d p) :
        id_(next_id_++), pose_{p}, n_obs_(0)
{
}

MapPoint::MapPoint(const Vector3d p, const KeyFrame::Ptr kf, const Feature::Ptr ft) :
        id_(next_id_++), pose_{p}, n_obs_(0)
{
    addObservation(kf, ft);
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
    std::unordered_map<KeyFrame::Ptr, Feature::Ptr>::iterator it = obs_.find(kf);
    if(it == obs_.end())
        return nullptr;
    else
        return it->second;
}

}
