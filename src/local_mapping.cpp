#include <include/config.hpp>
#include "local_mapping.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "optimizer.hpp"

namespace ssvo{

void showMatch(const cv::Mat &src, const cv::Mat &dst, const Vector2d &epl0, const Vector2d &epl1, const Vector2d &px_src, const Vector2d &px_dst)
{
    cv::Mat src_show = src.clone();
    cv::Mat dst_show = dst.clone();
    if(src_show.channels() == 1)
        cv::cvtColor(src_show, src_show, CV_GRAY2RGB);
    if(dst_show.channels() == 1)
        cv::cvtColor(dst_show, dst_show, CV_GRAY2RGB);

    cv::Point2d p0(epl0[0], epl0[1]);
    cv::Point2d p1(epl1[0], epl1[1]);
    cv::Point2d p_src(px_src[0], px_src[1]);
    cv::Point2d p_dst(px_dst[0], px_dst[1]);

    cv::line(dst_show, p0, p1, cv::Scalar(0, 255, 0), 1);
    cv::circle(dst_show, p1, 3, cv::Scalar(0,255,0));
    cv::circle(src_show, p_src, 5, cv::Scalar(255,0,0));
    cv::circle(dst_show, p_dst, 5, cv::Scalar(255,0,0));

    cv::imshow("ref", src_show);
    cv::imshow("dst", dst_show);
    cv::waitKey(0);
}

void showEplMatch(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame, const SE3d &T_cur_from_ref,
                  int level_ref, int level_cur, const Vector3d &xyz_near, const Vector3d &xyz_far, const Vector3d &xyz_ref, const Vector2d &px_dst)
{
    cv::Mat src_show = keyframe->getImage(level_ref).clone();
    cv::Mat dst_show = frame->getImage(level_cur).clone();
    if(src_show.channels() == 1)
        cv::cvtColor(src_show, src_show, CV_GRAY2RGB);
    if(dst_show.channels() == 1)
        cv::cvtColor(dst_show, dst_show, CV_GRAY2RGB);
    
    const double depth_cur = (T_cur_from_ref*xyz_ref)[2];
    Vector3d epl_near = xyz_near / xyz_near[2];
    Vector3d epl_far  = xyz_far / xyz_far[2];

    Vector2d epl0 = frame->cam_->project(epl_near);
    Vector2d epl1 = frame->cam_->project(epl_far);

    const double scale_cur = 1.0/(1<<level_cur);
    const double scale_ref = 1.0/(1<<level_ref);
    const Vector2d px_src = keyframe->cam_->project(xyz_ref);

    cv::Point2d pl0(epl0[0]*scale_cur, epl0[1]*scale_cur);
    cv::Point2d pl1(epl1[0]*scale_cur, epl1[1]*scale_cur);
    cv::Point2d p_src(px_src[0]*scale_ref, px_src[1]*scale_ref);
    cv::Point2d p_dst(px_dst[0], px_dst[1]);

    cv::line(dst_show, pl0, pl1, cv::Scalar(0, 255, 0), 1);
    cv::circle(dst_show, pl1, 2, cv::Scalar(0,0,255));
    cv::circle(src_show, p_src, 5, cv::Scalar(255,0,0));
    cv::circle(dst_show, p_dst, 5, cv::Scalar(255,0,0));

    //! ref
    SE3d T_ref_from_cur = T_cur_from_ref.inverse();
    Vector3d xyz_near_cur(epl_near * depth_cur);
    epl_far = T_ref_from_cur.translation();
    epl_near = xyz_ref;
    epl0 = keyframe->cam_->project(epl_near);
    epl1 = keyframe->cam_->project(epl_far);
    pl0.x = epl0[0]*scale_ref;
    pl0.y = epl0[1]*scale_ref;
    pl1.x = epl1[0]*scale_ref;
    pl1.y = epl1[1]*scale_ref;

    cv::line(src_show, pl0, pl1, cv::Scalar(0, 255, 0), 1);
    cv::circle(src_show, pl1, 2, cv::Scalar(0,0,255));

    cv::imshow("ref", src_show);
    cv::imshow("dst", dst_show);
    cv::waitKey(0);
}

void showAffine(const cv::Mat &src, const Vector2d &px_ref, const Matrix2d &A_ref_cur, const int size, const int level)
{
    cv::Mat src_show = src.clone();
    if(src_show.channels() == 1)
        cv::cvtColor(src_show, src_show, CV_GRAY2RGB);

    const double half_size = size*0.5;
    const int factor = 1 << level;
    Vector2d tl = A_ref_cur * Vector2d(-half_size, -half_size) * factor;
    Vector2d tr = A_ref_cur * Vector2d(half_size, -half_size) * factor;
    Vector2d bl = A_ref_cur * Vector2d(-half_size, half_size) * factor;
    Vector2d br = A_ref_cur * Vector2d(half_size, half_size) * factor;
    cv::Point2d TL(tl[0]+px_ref[0], tl[1]+px_ref[1]);
    cv::Point2d TR(tr[0]+px_ref[0], tr[1]+px_ref[1]);
    cv::Point2d BL(bl[0]+px_ref[0], bl[1]+px_ref[1]);
    cv::Point2d BR(br[0]+px_ref[0], br[1]+px_ref[1]);
    cv::Scalar color(0, 255, 0);
    cv::line(src_show, TL, TR, color, 1);
    cv::line(src_show, TR, BR, color, 1);
    cv::line(src_show, BR, BL, color, 1);
    cv::line(src_show, BL, TL, color, 1);
    cv::imshow("AFFINE", src_show);
    cv::waitKey(0);
}

//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report, bool verbose) :
    fast_detector_(fast_detector), delay_(static_cast<int>(1000.0/fps)), report_(report), verbose_(report&&verbose)
{
    map_ = Map::create();
    options_.max_kfs = 3;
    options_.max_epl_length = 10;
    options_.epl_dist2_threshold = 16 * Config::pixelUnSigma2();
    options_.seed_converge_threshold = 1.0/200.0;
    options_.klt_epslion = 0.0001;
    options_.align_epslion = 0.0001;
    options_.min_disparity = 30;
}

void LocalMapper::startThread()
{
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::stopThread()
{
    if(mapping_thread_ != nullptr)
    {
        setStop();
        mapping_thread_->join();
        mapping_thread_.reset();
    }
}

void LocalMapper::setStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    stop_require_ = true;
}

bool LocalMapper::isRequiredStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    return stop_require_;

}

void LocalMapper::run()
{
    while(!isRequiredStop())
    {
        if(!checkNewFrame())
            continue;

//        processNewFrame();
//
//        processNewKeyFrame();
    }
}

void LocalMapper::drowTrackedPoints(cv::Mat &dst)
{
    //! draw features
    const cv::Mat src = current_frame_->getImage(0);
    std::vector<Feature::Ptr> fts = current_frame_->getFeatures();
    cv::cvtColor(src, dst, CV_GRAY2RGB);
    int font_face = 1;
    double font_scale = 0.5;
    for(const Feature::Ptr &ft : fts)
    {
        Vector2d ft_px = ft->px;
        cv::Point2f px(ft_px[0], ft_px[1]);
        cv::Scalar color(0, 255, 0);
        cv::circle(dst, px, 2, color, -1);

        string id_str = std::to_string((current_frame_->Tcw()*ft->mpt->pose()).norm());
        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

    //! draw seeds
    for(const TrackSeed::Ptr &track_seed : tracked_seeds_)
    {
        cv::Point2f px = track_seed->px;
        double convergence = track_seed->seed->z_range/std::sqrt(track_seed->seed->sigma2);
        double scale = MIN(convergence, 256.0) / 256.0;
        cv::Scalar color(255*scale, 0, 255*(1-scale));
        cv::circle(dst, px, 2, color, -1);

//        string id_str = std::to_string();
//        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

}

void LocalMapper::insertNewFrame(const Frame::Ptr &frame)
{
    LOG_ASSERT(frame != nullptr) << "[Mapping] Error input! Frame should not be null!";
    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        last_frame_ = current_frame_;
        current_frame_ = frame;
        cond_process_.notify_one();
    }
    else
    {
        last_frame_ = current_frame_;
        current_frame_ = frame;
        const int num = trackSeeds();
        LOG_IF(INFO, report_) << "[Mapping][1] Tracking seeds: " << num;
    }
}

bool LocalMapper::finishFrame()
{
    int rest_num = updateSeeds();
    int reproj_num = reprojectSeeds();
    LOG_IF(INFO, report_) << "[Mapping][2] Seeds updated: " << rest_num << ", reprojected: " << reproj_num;
    return true;
}

void LocalMapper::insertNewKeyFrame(const KeyFrame::Ptr &keyframe, double mean_depth, double min_depth, bool optimal, bool create_seeds)
{
    LOG_ASSERT(keyframe != nullptr) << "Error input! Frame should not be null!";

    map_->insertKeyFrame(keyframe);
    const Features &fts = keyframe->features();
    for(const Feature::Ptr &ft : fts)
    {
        ft->mpt->addObservation(keyframe, ft);
        ft->mpt->updateViewAndDepth();
    }
    keyframe->updateConnections();

    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        current_keyframe_ = keyframe;
        current_depth_ = std::make_pair(mean_depth, min_depth);
        cond_process_.notify_one();
    }
    else
    {
        current_keyframe_ = keyframe;
        current_depth_ = std::make_pair(mean_depth, min_depth);

        int new_features = createNewFeatures();

        if(optimal)
            Optimizer::localBundleAdjustment(current_keyframe_, true, true);

        int new_seeds = 0;
        if(create_seeds)
            new_seeds = createSeeds();

        LOG_IF(INFO, report_) << "[Mapping] New created features:" << new_features << ", seeds: " << new_seeds;
    }
}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = KeyFrame::create(frame_cur);

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
        mpt->resetType(MapPoint::STABLE);

        mpt->addObservation(keyframe_ref, fts_ref[i]);
        mpt->addObservation(keyframe_cur, fts_cur[i]);
        mpt->updateViewAndDepth();
    }

    Vector2d mean_depth, min_depth;
    keyframe_ref->getSceneDepth(mean_depth[0], min_depth[0]);
    keyframe_cur->getSceneDepth(mean_depth[1], min_depth[1]);
    this->insertNewKeyFrame(keyframe_ref, mean_depth[0], min_depth[0], false, false);
    this->insertNewKeyFrame(keyframe_cur, mean_depth[1], min_depth[1], false);

    //! set current frame and create inital seeds flow
    current_frame_ = frame_cur;

    LOG_IF(INFO, report_) << "[Mapping] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

bool LocalMapper::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        cond_process_.wait_for(lock, std::chrono::milliseconds(delay_));
        if(frames_buffer_.empty())
            return false;

//        current_frame_ = frames_buffer_.front();
//        current_depth_ = depth_buffer_.front();
//        frames_buffer_.pop_front();
//        depth_buffer_.pop_front();
    }

    return true;
}

int LocalMapper::createSeeds(bool is_track)
{
    if(current_keyframe_ == nullptr)
        return 0;

    std::vector<Feature::Ptr> fts = current_keyframe_->getFeatures();

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    // TODO 如果对应的seed收敛，在跟踪过的关键帧增加观测？
    if(current_frame_ != nullptr && current_keyframe_->frame_id_ == current_frame_->id_)
    {
        for(const TrackSeed::Ptr &seed : tracked_seeds_)
        {
            const cv::Point2f px = seed->px;
            old_corners.emplace_back(Corner(px.x, px.y, 0, seed->seed->ft->level));
        }
    }

    Corners new_corners;
    fast_detector_->detect(current_keyframe_->images(), new_corners, old_corners, 150);

    if(new_corners.empty())
        return 0;

    Seeds new_seeds;
    for(const Corner &corner : new_corners)
    {
        const Vector2d px(corner.x, corner.y);
        const Vector3d fn(current_keyframe_->cam_->lift(px));
        Feature::Ptr ft = Feature::create(px, fn, corner.level, nullptr);
        new_seeds.emplace_back(Seed::create(ft, current_depth_.first, current_depth_.second*0.5));
    }
    seeds_buffer_.emplace_back(current_keyframe_, std::make_shared<Seeds>(new_seeds));

    if(is_track)
    {
        const auto &seed_to_track = seeds_buffer_.back();
        for(const Seed::Ptr &seed : *seed_to_track.second)
        {
            const cv::Point2f px(seed->ft->px[0], seed->ft->px[1]);
            tracked_seeds_.emplace_back(std::make_shared<TrackSeed>(TrackSeed(current_keyframe_, seed, px)));
        }
    }

    return (int)new_seeds.size();
}

int LocalMapper::createNewFeatures()
{
    std::map<KeyFrame::Ptr, std::deque<TrackSeed::Ptr> > seeds_map;
    for(const TrackSeed::Ptr &track_seed : tracked_seeds_)
    {
        const auto seeds_map_itr = seeds_map.lower_bound(track_seed->kf);
        if(seeds_map_itr == seeds_map.end())
        {
            std::deque<TrackSeed::Ptr> new_deque;
            new_deque.push_back(track_seed);
            seeds_map.emplace(track_seed->kf, new_deque);
        }
        else
        {
            seeds_map_itr->second.push_back(track_seed);
        }
    }

    int count = 0;
    tracked_seeds_.clear();
    for(const auto &seed_map : seeds_map)
    {
        KeyFrame::Ptr kf = seed_map.first;
        const std::deque<TrackSeed::Ptr> &seeds_deque = seed_map.second;
        const SE3d T_cur_from_ref = current_frame_->Tcw() * kf->pose();
        for(const TrackSeed::Ptr &track_seed : seeds_deque)
        {
            Vector2d px_cur(track_seed->px.x, track_seed->px.y);
            double disparity = (px_cur - track_seed->seed->ft->px).norm();
            if(disparity > options_.min_disparity)
            {
                double depth;
                const Vector3d fn_ref = track_seed->seed->ft->fn;
                const Vector3d fn_cur = current_frame_->cam_->lift(px_cur);
                bool succeed = utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), fn_ref, fn_cur, depth);
                if(succeed)
                {
                    Feature::Ptr ref_ft = track_seed->seed->ft;
                    Vector3d pose(fn_ref*depth);
                    MapPoint::Ptr new_mpt = MapPoint::create(kf->pose()*pose, current_keyframe_);
                    //! ref kf
                    ref_ft->mpt = new_mpt;
                    kf->addFeature(ref_ft);
                    //! cur kf
                    Feature::Ptr new_ft = Feature::create(px_cur, fn_cur, ref_ft->level, new_mpt);
                    current_keyframe_->addFeature(new_ft);

                    //! mpt
                    map_->insertMapPoint(new_mpt);
                    new_mpt->addObservation(kf, ref_ft);
                    new_mpt->addObservation(current_keyframe_, new_ft);
                    new_mpt->updateViewAndDepth();
//                    Optimizer::refineMapPoint(new_mpt, 20, true, true);
                    count++;
                    continue;
                }
            }

            tracked_seeds_.push_back(track_seed);
        }
    }

    return count;
}


int LocalMapper::trackSeeds()
{
    if(current_frame_ == nullptr || last_frame_ == nullptr)
        return 0;

    //! track seeds by klt
    std::vector<cv::Point2f> pts_to_track;
    pts_to_track.reserve(tracked_seeds_.size());
    for(const TrackSeed::Ptr &track_seed : tracked_seeds_)
    {
        pts_to_track.emplace_back(track_seed->px);
    }

    if(pts_to_track.empty())
        return 0;

    std::vector<cv::Point2f> pts_tracked = pts_to_track;
    std::vector<bool> status;
    static cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, options_.klt_epslion);
    utils::kltTrack(last_frame_->opticalImages(), current_frame_->opticalImages(), Frame::optical_win_size_,
                    pts_to_track, pts_tracked, status, termcrit, true, verbose_);

    //! erase untracked seeds
    size_t idx = 0;
    int count = 0;
    auto track_seeds_itr = tracked_seeds_.begin();
    for(;track_seeds_itr!=tracked_seeds_.end(); idx++)
    {
        if(status[idx])
        {
            (*track_seeds_itr)->px = pts_tracked[idx];
            track_seeds_itr++;
            count++;
        }
        else
        {
            track_seeds_itr = tracked_seeds_.erase(track_seeds_itr);
        }
    }

    return count;
}

int LocalMapper::updateSeeds()
{
    //! remove error tracked seeds and update
    std::map<KeyFrame::Ptr, std::deque<TrackSeed::Ptr> > seeds_map;
    for(const TrackSeed::Ptr &track_seed : tracked_seeds_)
    {
        const auto seeds_map_itr = seeds_map.lower_bound(track_seed->kf);
        if(seeds_map_itr == seeds_map.end())
        {
            std::deque<TrackSeed::Ptr> new_deque;
            new_deque.push_back(track_seed);
            seeds_map.emplace(track_seed->kf, new_deque);
        }
        else
        {
            seeds_map_itr->second.push_back(track_seed);
        }
    }

    static double px_error_angle = atan(0.5*Config::pixelUnSigma())*2.0;
    tracked_seeds_.clear();
    for(const auto &seed_map : seeds_map)
    {
        KeyFrame::Ptr kf = seed_map.first;
        const std::deque<TrackSeed::Ptr> &seeds_deque = seed_map.second;
        const SE3d T_cur_from_ref = current_frame_->Tcw() * kf->pose();
        const SE3d T_ref_from_cur = T_cur_from_ref.inverse();
        for(const TrackSeed::Ptr &track_seed : seeds_deque)
        {
            const Vector3d fn_ref = track_seed->seed->ft->fn;
            const Vector3d fn_cur = current_frame_->cam_->lift(track_seed->px.x, track_seed->px.y);
            double err2 = utils::Fundamental::computeErrorSquared(
                kf->pose().translation(), fn_ref/track_seed->seed->mu, T_cur_from_ref, fn_cur.head<2>());

            if(err2 > options_.epl_dist2_threshold)
                continue;

            //! update
            double depth = -1;
            if(utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), fn_ref, fn_cur, depth))
            {
                double tau = track_seed->seed->computeTau(T_ref_from_cur, fn_ref, depth, px_error_angle);
//                double tau = track_seed->seed->computeVar(T_cur_from_ref, depth, options_.klt_epslion);
                tau = tau + Config::pixelUnSigma();
                track_seed->seed->update(1.0/depth, tau*tau);

                //! check converge
                if(track_seed->seed->sigma2 / track_seed->seed->z_range < options_.seed_converge_threshold)
                {
                    createFeatureFromSeed(track_seed->kf ,track_seed->seed);
                    earseSeed(track_seed->kf ,track_seed->seed);
                    continue;
                }
            }

            tracked_seeds_.emplace_back(track_seed);
        }
    }

    return (int)tracked_seeds_.size();
}

int LocalMapper::reprojectSeeds()
{
    //! get new seeds for track
    std::set<KeyFrame::Ptr> candidate_keyframes = current_frame_->getRefKeyFrame()->getConnectedKeyFrames(options_.max_kfs);
    candidate_keyframes.insert(current_frame_->getRefKeyFrame());

    std::set<Seed::Ptr> seed_tracking;
    for(const TrackSeed::Ptr &tracked_seed : tracked_seeds_)
        seed_tracking.insert(tracked_seed->seed);

    int count = 0;
    static double px_error_angle = atan(0.5*Config::pixelUnSigma())*2.0;
    auto buffer_itr = seeds_buffer_.begin();
    for(;buffer_itr != seeds_buffer_.end();)
    {
        KeyFrame::Ptr keyframe = buffer_itr->first;
        if(!candidate_keyframes.count(keyframe))
        {
            buffer_itr++;
            continue;
        }

        SE3d T_cur_from_ref = current_frame_->Tcw() * keyframe->pose();
        SE3d T_ref_from_cur = T_cur_from_ref.inverse();
        Seeds &seeds = *buffer_itr->second;

        Vector2d px_matched;
        int level_matched;
        auto seed_itr = seeds.begin();
        for(; seed_itr != seeds.end();)
        {
            const Seed::Ptr &seed = *seed_itr;
            if(seed_tracking.count(seed))
            {
                seed_itr++;
                continue;
            }

            bool matched = findEpipolarMatch(seed, keyframe, current_frame_, T_cur_from_ref, px_matched, level_matched);
            if(!matched)
            {
                seed_itr++;
                continue;
            }

            //! check distance to epl, incase of the aligen draft
            Vector2d fn_matched = current_frame_->cam_->lift(px_matched).head<2>();
            double dist2 = utils::Fundamental::computeErrorSquared(keyframe->pose().translation(), seed->ft->fn/seed->mu, T_cur_from_ref, fn_matched);
            if(dist2 > options_.epl_dist2_threshold)
            {
                seed_itr++;
                continue;
            }

            double depth = -1;
            const Vector3d fn_cur = current_frame_->cam_->lift(px_matched);
            bool succeed = utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), seed->ft->fn, fn_cur, depth);
            if(!succeed)
            {
                seed_itr++;
                continue;
            }

            double tau = seed->computeTau(T_ref_from_cur, seed->ft->fn, depth, px_error_angle);
//            double tau = seed->computeVar(T_cur_from_ref, depth, options_.align_epslion);
            tau = tau + Config::pixelUnSigma();
            seed->update(1.0/depth, tau*tau);

            //! check converge
            if(seed->sigma2 / seed->z_range < options_.seed_converge_threshold)
            {
                createFeatureFromSeed(keyframe, seed);
                seed_itr = seeds.erase(seed_itr);
                continue;
            }

            const cv::Point2f px(px_matched[0], px_matched[1]);
            tracked_seeds_.emplace_back(std::make_shared<TrackSeed>(TrackSeed(keyframe, seed, px)));
            count++;
            seed_itr++;
        }

        if(seeds.empty())
            buffer_itr = seeds_buffer_.erase(buffer_itr);
        else
            buffer_itr++;
    }

    return count;
}


bool LocalMapper::earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed)
{
    //! earse seed
    auto buffer_itr = seeds_buffer_.begin();
    for(; buffer_itr != seeds_buffer_.end(); buffer_itr++)
    {
        if(keyframe != buffer_itr->first)
            continue;

        Seeds &seeds = *buffer_itr->second;
        auto seeds_itr = seeds.begin();
        for(; seeds_itr != seeds.end(); seeds_itr++)
        {
            if(seed != *seeds_itr)
                continue;

            seeds_itr = seeds.erase(seeds_itr);
            if(seeds.empty())
                seeds_buffer_.erase(buffer_itr);

            return true;
        }
    }

    return false;
}

bool LocalMapper::createFeatureFromSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed)
{
    //! create new feature
    // TODO add_observation 不用，需不需要找一下其他关键帧是否观测改点，增加约束
    // TODO 把这部分放入一个队列，在单独线程进行处理
    Feature::Ptr new_ft = seed->ft;
    Vector3d pose(new_ft->fn/seed->mu);
    new_ft->mpt = MapPoint::create(keyframe->pose() * pose, keyframe);
    keyframe->addFeature(new_ft);
    map_->insertMapPoint(new_ft->mpt);
    new_ft->mpt->addObservation(keyframe, new_ft);
    new_ft->mpt->updateViewAndDepth();
    std::cout << " Creat3 new seed as mpt: " << new_ft->mpt->id_ << ", " << 1.0/seed->mu << ", kf: " << keyframe->id_ << " his: ";
    for(const auto his : seed->history){ std::cout << "[" << his.first << "," << his.second << "]";}
    std::cout << std::endl;

    return true;
}

//bool LocalMapper::processNewKeyFrame()
//{
//    if(current_frame_.second == nullptr)
//        return false;
//
//    const KeyFrame::Ptr &kf = current_frame_.second;
//    const double mean_depth = current_depth_.first;
//    const double min_depth = current_depth_.second;
//
//    //! check base line
//    for(auto &seeds_pair : seeds_buffer_)
//    {
//        const KeyFrame::Ptr ref_kf = seeds_pair.first;
//        Seeds &seeds = *seeds_pair.second;
//
//        SE3d T_ref_cur =  ref_kf->Tcw() * kf->pose();
//
//        double base_line = T_ref_cur.translation().norm();
//        std::cout << "b: " << base_line << "d: " << mean_depth << " " << min_depth << std::endl;
//    }
//
//
//    std::vector<Feature::Ptr> fts = kf->getFeatures();
//
//    Corners old_corners;
//    old_corners.reserve(fts.size());
//    for(const Feature::Ptr &ft : fts)
//    {
//        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
//    }
//
//    Corners new_corners;
//    fast_detector_->detect(kf->image(), new_corners, old_corners, 150);
//
//    Seeds new_seeds;
//    for(const Corner &corner : new_corners)
//    {
//        const Vector2d px(corner.x, corner.y);
//        const Vector3d fn(kf->cam_->lift(px));
//        Feature::Ptr ft = Feature::create(px, fn, corner.level, nullptr);
//        new_seeds.emplace_back(Seed::create(ft, mean_depth, min_depth*0.5));
//    }
//    seeds_buffer_.emplace_back(kf, std::make_shared<Seeds>(new_seeds));
//
//    // TODO
//    if(seeds_buffer_.size() > 5)
//        seeds_buffer_.pop_front();
//
//    LOG_IF(WARNING, report_) << "[Mapping] Add new keyframe " << kf->id_ << " with " << new_seeds.size() << " seeds";
//
//    return true;
//}
//
//bool LocalMapper::processNewFrame()
//{
//    if(current_frame_.first == nullptr)
//        return false;
//
//    const Frame::Ptr &frame = current_frame_.first;
//
//    double t0 = (double)cv::getTickCount();
//    double px_error_angle = atan(0.5*Config::pixelUnSigma())*2.0;
//    for(auto &seeds_pair : seeds_buffer_)
//    {
//        const KeyFrame::Ptr keyframe = seeds_pair.first;
//        Seeds &seeds = *seeds_pair.second;
//
//        // TODO old seeds_pair remove
//
//        SE3d T_cur_from_ref = frame->Tcw() * keyframe->pose();
//        SE3d T_ref_from_cur = T_cur_from_ref.inverse();
//        for(auto it = seeds.begin(); it!=seeds.end(); it++)
//        {
//            const Seed::Ptr &seed = *it;
//            double depth = 1.0/seed->mu;
//            if(!findEpipolarMatch(seed->ft, keyframe, frame, T_cur_from_ref, seed->sigma2, depth))
//                continue;
//
//            double tau = seed->computeTau(T_ref_from_cur, seed->ft->fn, depth, px_error_angle);
//            seed->update(1.0/depth, tau*tau);
//        }
//    }
//
//    double t1 = (double)cv::getTickCount();
//    if(!seeds_buffer_.empty())
//        LOG_IF(INFO, report_) << "Seeds update time: " << (t1-t0)/cv::getTickFrequency()/seeds_buffer_.size() << " * " << seeds_buffer_.size();
//
//    return true;
//}

bool LocalMapper::findEpipolarMatch(const Seed::Ptr &seed,
                                    const KeyFrame::Ptr &keyframe,
                                    const Frame::Ptr &frame,
                                    const SE3d &T_cur_from_ref,
                                    Vector2d &px_matched,
                                    int &level_matched)
{
    static const int patch_size = Align2DI::PatchSize;
    static const int patch_area = Align2DI::PatchArea;
    static const int half_patch_size = Align2DI::HalfPatchSize;

    //! check if in the view of current frame
    const Feature::Ptr &ft = seed->ft;
    const double z_ref = 1.0/seed->mu;
    const Vector3d xyz_ref(ft->fn * z_ref);
    const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    const double z_cur = xyz_cur[2];
    if(z_cur < 0.00f)
        return false;

    const Vector2d px_cur(frame->cam_->project(xyz_cur));
    if(!frame->cam_->isInFrame(px_cur.cast<int>(), half_patch_size))
        return false;

    //! d - inverse depth, z - depth
    const double sigma = std::sqrt(seed->sigma2);
    const double d_max = z_ref + sigma;
    const double d_min = MAX(z_ref- sigma, 0.00000001f);
    const double z_ref_min = 1.0/d_max;
    const double z_ref_max = 1.0/d_min;

    //! on unit plane in cur frame
    Vector3d xyz_near = T_cur_from_ref * (ft->fn * z_ref_min);
    Vector3d xyz_far  = T_cur_from_ref * (ft->fn * z_ref_max);
    xyz_near /= xyz_near[2];
    xyz_far  /= xyz_far[2];
    Vector2d epl_dir = (xyz_near - xyz_far).head<2>();

    //! calculate best search level
    const int level_ref = ft->level;
    const int level_cur = MapPoint::predictScale(z_ref, z_cur, level_ref, frame->max_level_);
    const double factor = static_cast<double>(1 << level_cur);

    //! px in image plane
    Vector2d px_near = frame->cam_->project(xyz_near) / factor;
    Vector2d px_far  = frame->cam_->project(xyz_far) / factor;
    if(!frame->cam_->isInFrame(px_far.cast<int>(), half_patch_size, level_cur) ||
       !frame->cam_->isInFrame(px_near.cast<int>(), half_patch_size, level_cur))
        return false;

    //! reject the seed whose epl is too long
    double epl_length = (px_near-px_far).norm();
    if(epl_length > options_.max_epl_length)
        return false;

    //! get warp patch
    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(keyframe->cam_, frame->cam_, ft->px, ft->fn, level_ref,
                               z_ref, T_cur_from_ref, patch_size, A_cur_from_ref);

//    double det = A_cur_from_ref.determinant() / factor;
//    std::cout << "***** det: " <<  det << std::endl;

    static const int patch_border_size = patch_size+2;
    cv::Mat image_ref = keyframe->getImage(level_ref);
    Matrix<double, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<double, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                 ft->px, level_ref, level_cur);

    Matrix<double, patch_size, patch_size, RowMajor> patch;
    patch = patch_with_border.block(1, 1, patch_size, patch_size);

    const cv::Mat image_cur = frame->getImage(level_cur);
    Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> > image_cur_eigen((uchar*)image_cur.data, image_cur.rows, image_cur.cols);
    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > patch_eigen(patch.data(), patch_area, 1);

    Vector2d px_best(-1,-1);
    if(epl_length > 2.0)
    {
        int n_steps = epl_length / 0.707;
        Vector2d step = epl_dir / n_steps;

        // TODO 使用模板来加速！！！
        //! SSD
        double t0 = (double)cv::getTickCount();
        ZSSD<double, patch_area> zssd(patch_eigen);
        double score_best = std::numeric_limits<double>::max();
        double score_second = score_best;
        int index_best = -1;
        int index_second = -1;

        Vector2d fn_start = xyz_far.head<2>() - step * 2;
        n_steps += 2;
        Vector2d fn(fn_start);
        for(int i = 0; i < n_steps; ++i, fn += step)
        {
            Vector2d px(frame->cam_->project(fn) / factor);

            //! always in frame's view
            if(!frame->cam_->isInFrame(px.cast<int>(), half_patch_size, level_cur))
                continue;

            Matrix<double, patch_area, 1> patch_cur;
            utils::interpolateMat<uchar, double, patch_size>(image_cur_eigen, patch_cur, px[0], px[1]);

            double score = zssd.compute_score(patch_cur);

            if(score < score_best)
            {
                score_best = score;
                index_best = i;
            }
            else if(score < score_second)
            {
                score_second = score;
                index_second = i;
            }
        }

        if(score_best > 0.8 * score_second && std::abs(index_best - index_second) > 2)
            return false;

        if(score_best > zssd.threshold())
            return false;

        Vector2d pn_best = fn_start + index_best * step;
        px_best = frame->cam_->project(pn_best) / factor;

        double t1 = (double)cv::getTickCount();
        double time = (t1-t0)/cv::getTickFrequency();
        LOG_IF(INFO, verbose_) << " Step: " << n_steps << " T:" << time << "(" << time/n_steps << ")"
                               << " best: [" << index_best << ", " << score_best<< "]"
                               << " second: [" << index_second << ", " << score_second<< "]";
    }
    else
    {
        px_best = (px_near + px_far) * 0.5;
    }

    Matrix<double, patch_size, patch_size, RowMajor> dx, dy;
    dx = 0.5*(patch_with_border.block(1, 2, patch_size, patch_size)
        - patch_with_border.block(1, 0, patch_size, patch_size));
    dy = 0.5*(patch_with_border.block(2, 1, patch_size, patch_size)
        - patch_with_border.block(0, 1, patch_size, patch_size));

    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dx_eigen(dx.data(), patch_area, 1);
    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dy_eigen(dy.data(), patch_area, 1);

    Align2DI matcher(verbose_);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_best;
    if(!matcher.run(image_cur_eigen, patch_eigen, dx_eigen, dy_eigen, estimate, 30, options_.align_epslion))
    {
//        std::cout << "dx:\n " << dx << std::endl;
//        std::cout << "dy:\n " << dy << std::endl;

        static bool show = false;
        if(show)
        {
            //        showMatch(keyframe->getImage(level_ref), current_frame_->getImage(level_cur), px_near, px_far, ft->px/factor, px_best);
            showEplMatch(keyframe, frame, T_cur_from_ref, level_ref, level_cur, xyz_near, xyz_far, xyz_ref, px_best);
            showAffine(keyframe->getImage(level_ref), ft->px/factor, A_cur_from_ref.inverse(), 8, level_ref);
        }

        return false;
    }

    //! transform to level-0
    px_matched = estimate.head<2>() * factor;

    LOG_IF(INFO, verbose_) << "Found! [" << ft->px.transpose() << "] "
                           << "dst: [" << px_matched.transpose() << "] "
                           << "epl: [" << px_near.transpose() << "]--[" << px_far.transpose() << "]" << std::endl;

//    showMatch(image_ref, image_cur, px_near, px_far, seed.ft->px/factor, px_matched/factor);

    return true;
}


}