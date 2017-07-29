#include "map.hpp"

namespace ssvo{

void Map::insertKeyFrame(const KeyFrame::Ptr kf)
{
    kfs_.insert(std::make_pair(kf->id(), kf));
}

void Map::deleteKeyFrame(const KeyFrame::Ptr kf)
{
    kfs_.erase(kf->id());
}

void Map::insertMapPoint(const MapPoint::Ptr mpt)
{
    mpts_.insert(std::make_pair(mpt->id(), mpt));
}

void Map::deleteMapPoint(const MapPoint::Ptr mpt)
{
    mpts_.erase(mpt->id());
}

}