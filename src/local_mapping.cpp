#include "local_mapping.hpp"

namespace ssvo{

LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, const Map::Ptr &map, double fps, bool report) :
    fast_detector_(fast_detector), map_(map), delay_(static_cast<int>(1000.0/fps)), report_(report)
{
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::insertNewKeyFrame(const KeyFrame::Ptr &kf)
{
    std::unique_lock<std::mutex> lock(mutex_kfs_);
    new_keyframes_.push_back(kf);
    cond_kfs_.notify_one();
}

void LocalMapper::run()
{
    while(true)
    {
        {
            std::unique_lock<std::mutex> lock(mutex_kfs_);
            cond_kfs_.wait_for(lock, std::chrono::milliseconds(delay_));
            if(new_keyframes_.empty())
                continue;
            current_keyframe_ = new_keyframes_.front();
            new_keyframes_.pop_front();
        }

        LOG(INFO) << "[Mapping] Add new keyframe " << current_keyframe_->id_;
    }
}

}