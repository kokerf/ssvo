#include "local_mapping.hpp"

namespace ssvo{

LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, const Map::Ptr &map, double fps, bool report) :
    fast_detector_(fast_detector), map_(map), delay_(static_cast<int>(1000.0/fps)), report_(report)
{
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::insertNewKeyFrame(const KeyFrame::Ptr &kf, double mean_depth, double min_depth)
{
    std::unique_lock<std::mutex> lock(mutex_kfs_);
    new_keyframes_.emplace_back(KFD(kf, mean_depth, min_depth));
    cond_kfs_.notify_one();
}

void LocalMapper::run()
{
    while(true)
    {
        processNewKeyFrame();

        createNewMapPoints();
    }
}

void LocalMapper::processNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        cond_kfs_.wait_for(lock, std::chrono::milliseconds(delay_));
        if(new_keyframes_.empty())
            return;
        keyframe_depth_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }

    const KeyFrame::Ptr &kf = keyframe_depth_.kf;
    fts_ = kf->getFeatures();

    for(const Feature::Ptr &ft : fts_)
    {
        MapPoint::Ptr mpt = ft->mpt;
        if(mpt == nullptr)
            continue;

        mpt->addObservation(kf, ft);
        mpt->updateViewAndDepth();
    }

    kf->updateConnections();
    map_->insertKeyFrame(kf);

    LOG(INFO) << "[Mapping] Add new keyframe " << kf->id_;
}

void LocalMapper::createNewMapPoints()
{
    Corners old_corners;
    old_corners.reserve(fts_.size());
    for(Feature::Ptr ft : fts_)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    const KeyFrame::Ptr &kf = keyframe_depth_.kf;
    Corners new_corners;
    fast_detector_->detect(kf->image(), new_corners, old_corners, 150);
}

}