#include "map_point.hpp"

namespace ssvo
{

unsigned long int MapPoint::next_id_ = 0;

MapPoint::MapPoint(const Vector3d p) :
        id_(next_id_++), pose_{p}, n_obs_(0)
{
}

MapPoint::MapPoint(const Vector3d p, const std::shared_ptr<KeyFrame> kf, const Feature::Ptr ft) :
        id_(next_id_++), pose_{p}, n_obs_(0)
{
    addObservation(kf, ft);
}

void MapPoint::addObservation(const std::shared_ptr<KeyFrame> kf, const Feature::Ptr ft)
{
    obs_.insert(std::make_pair(kf, ft));
    n_obs_++;
}

Feature::Ptr MapPoint::findObservation(const std::shared_ptr<KeyFrame> kf)
{

    std::unordered_map<std::shared_ptr<KeyFrame>, Feature::Ptr>::iterator it = obs_.find(kf);
    if(it == obs_.end())
        return nullptr;
    else
        return it->second;
}

}
