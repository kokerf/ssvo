#include "map.hpp"

namespace ssvo{

void Map::clear()
{
    std::lock_guard<std::mutex> lock_kf(mutex_kf_);
    std::lock_guard<std::mutex> lock_mpt(mutex_mpt_);
    kfs_.clear();
    mpts_.clear();
}

void Map::insertKeyFrame(const KeyFrame::Ptr &kf)
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    kfs_.emplace(kf->id_, kf);
}

void Map::removeKeyFrame(const KeyFrame::Ptr &kf)
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    kfs_.erase(kf->id_);
}

void Map::insertMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    mpts_.emplace(mpt->id_, mpt);
}

void Map::removeMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    mpts_.erase(mpt->id_);
    removed_mpts_.push_back(mpt);
//    std::string log;
//    log += "removed mpts: [ ";
//    for(const MapPoint::Ptr &rm_mpt : removed_mpts_)
//    {
//        log += "(" + std::to_string(rm_mpt->id_) + ", " + std::to_string(rm_mpt.use_count()) + "), ";
//    }
//    log += "] " + std::to_string(removed_mpts_.size());
//    LOG(INFO) << log;
}

std::vector<KeyFrame::Ptr> Map::getAllKeyFrames()
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    std::vector<KeyFrame::Ptr> kfs;
    kfs.reserve(kfs_.size());
    for(const auto &kf : kfs_)
        kfs.push_back(kf.second);

    return kfs;
}

KeyFrame::Ptr Map::getKeyFrame(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_kf_);
    if(kfs_.count(id))
        return kfs_[id];
    else
        return nullptr;
}

std::vector<MapPoint::Ptr> Map::getAllMapPoints()
{
    std::lock_guard<std::mutex> lock(mutex_mpt_);
    std::vector<MapPoint::Ptr> mpts;
    mpts.reserve(mpts_.size());
    for(const auto &mpt : mpts_)
        mpts.push_back(mpt.second);

    return mpts;
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