#include "map.hpp"

namespace ssvo{

void Map::clear()
{
    std::lock_guard<std::mutex> lock_kf(mutex_kf_);
    std::lock_guard<std::mutex> lock_mpt(mutex_mpt_);
    kfs_.clear();
    mpts_.clear();
}

void Map::insertKeyFrame(const KeyFrame::Ptr kf)
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    kfs_.insert(kf);
}

void Map::removeKeyFrame(const KeyFrame::Ptr kf)
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    kfs_.erase(kf);
}

void Map::insertMapPoint(const MapPoint::Ptr mpt)
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    mpts_.insert(mpt);
}

void Map::removeMapPoint(const MapPoint::Ptr mpt)
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    mpts_.erase(mpt);
}

std::vector<KeyFrame::Ptr> Map::getAllKeyFrames()
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    return std::vector<KeyFrame::Ptr>(kfs_.begin(), kfs_.end());
}

std::vector<KeyFrame::Ptr> Map::getAllKeyFramesOrderedByID()
{
    std::vector<KeyFrame::Ptr> keyframe_order_by_id;

    {
        std::lock_guard<std::mutex> lock(mutex_kf_);
        keyframe_order_by_id = std::vector<KeyFrame::Ptr>(kfs_.begin(), kfs_.end());
    }

    std::sort(keyframe_order_by_id.begin(), keyframe_order_by_id.end(), [](KeyFrame::Ptr kf1, KeyFrame::Ptr kf2){
        return kf1->id_ < kf2->id_;
    });

    return keyframe_order_by_id;
}

std::vector<MapPoint::Ptr> Map::getAllMapPoints()
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    return std::vector<MapPoint::Ptr>(mpts_.begin(), mpts_.end());
}

uint64_t Map::KeyFramesInMap()
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    return kfs_.size();
}

uint64_t Map::MapPointsInMap()
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    return mpts_.size();
}

}