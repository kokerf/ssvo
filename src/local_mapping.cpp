#include "local_mapping.hpp"

namespace ssvo{

LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report) :
    fast_detector_(fast_detector), delay_(static_cast<int>(1000.0/fps)), report_(report)
{
    map_ = Map::create();
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = ssvo::KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = ssvo::KeyFrame::create(frame_cur);

    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    LOG_ASSERT(N == points.size()) << "Error in create inital map! Two frames' features is not matched mappoints!";
    for(size_t i = 0; i < N; i++)
    {
        MapPoint::Ptr mpt = ssvo::MapPoint::create(points[i], keyframe_ref);

        fts_ref[i]->mpt = mpt;
        fts_cur[i]->mpt = mpt;

        map_->insertMapPoint(mpt);

        mpt->addObservation(keyframe_ref, fts_ref[i]);
        mpt->addObservation(keyframe_cur, fts_cur[i]);
        mpt->updateViewAndDepth();
    }

    Vector2d mean_depth, min_depth;
    keyframe_ref->getSceneDepth(mean_depth[0], min_depth[0]);
    keyframe_cur->getSceneDepth(mean_depth[1], min_depth[1]);
    this->insertNewFrame(frame_ref, keyframe_ref, mean_depth[0], min_depth[0]);
    this->insertNewFrame(frame_cur, keyframe_cur, mean_depth[1], min_depth[1]);

    size_t n = map_->MapPointsInMap();

    LOG_IF(INFO, report_) << "[Mapping] Creating inital map with " << n << " map points";
}

void LocalMapper::insertNewFrame(Frame::Ptr frame, KeyFrame::Ptr keyframe, double mean_depth, double min_depth)
{
    map_->insertKeyFrame(keyframe);
    keyframe->updateConnections();

    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);

        frames_buffer_.emplace_back(frame, keyframe);
        depth_buffer_.emplace_back(mean_depth, min_depth);
        cond_process_.notify_one();
    }
}
void LocalMapper::run()
{
    while(true)
    {
        if(!checkNewFrame())
            continue;

        processNewFrame();

        processNewKeyFrame();
    }
}

bool LocalMapper::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        cond_process_.wait_for(lock, std::chrono::milliseconds(delay_));
        if(frames_buffer_.empty())
            return false;

        current_frame = frames_buffer_.front();
        current_depth = depth_buffer_.front();
        frames_buffer_.pop_front();
        depth_buffer_.pop_front();
    }

    return true;
}

bool LocalMapper::processNewKeyFrame()
{
    if(current_frame.second == nullptr)
        return false;

    const KeyFrame::Ptr &kf = current_frame.second;
    std::vector<Feature::Ptr> fts = kf->getFeatures();

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(Feature::Ptr ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    Corners new_corners;
    fast_detector_->detect(kf->image(), new_corners, old_corners, 150);



    LOG(INFO) << "[Mapping] Add new keyframe " << kf->id_;

    return true;
}

bool LocalMapper::processNewFrame()
{
    if(current_frame.first == nullptr)
        return false;

    return true;
}


}