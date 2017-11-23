#include "map.hpp"

namespace ssvo{

void Map::clear()
{
    kfs_.clear();
    mpts_.clear();
}

void Map::insertKeyFrame(const KeyFrame::Ptr kf)
{
    kfs_.insert(kf);
}

void Map::removeKeyFrame(const KeyFrame::Ptr kf)
{
    kfs_.erase(kf);
}

void Map::insertMapPoint(const MapPoint::Ptr mpt)
{
    mpts_.insert(mpt);
}

void Map::removeMapPoint(const MapPoint::Ptr mpt)
{
    mpts_.erase(mpt);
}

std::vector<KeyFrame::Ptr> Map::getAllKeyFrames()
{
    return std::vector<KeyFrame::Ptr>(kfs_.begin(), kfs_.end());
}

std::vector<KeyFrame::Ptr> Map::getAllKeyFramesOrderedByID()
{
    std::vector<KeyFrame::Ptr> keyframe_order_by_id(kfs_.begin(), kfs_.end());
    std::sort(keyframe_order_by_id.begin(), keyframe_order_by_id.end(), [](KeyFrame::Ptr kf1, KeyFrame::Ptr kf2){
        return kf1->id_ < kf2->id_;
    });

    return keyframe_order_by_id;
}

std::vector<MapPoint::Ptr> Map::getAllMapPoints()
{
    return std::vector<MapPoint::Ptr>(mpts_.begin(), mpts_.end());
}

}