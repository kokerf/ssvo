#include <include/config.hpp>
#include "local_mapping.hpp"
#include "feature_alignment.hpp"
#include "feature_tracker.hpp"
#include "image_alignment.hpp"
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

//! =================================================================================================
//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report, bool verbose) :
    fast_detector_(fast_detector), delay_(static_cast<int>(1000.0/fps)), report_(report), verbose_(report&&verbose), status_track_thread_(true)
{
    map_ = Map::create();
    options_.max_kfs = 3;
    options_.max_epl_length = 1000;
    options_.epl_dist2_threshold = 16 * Config::pixelUnSigma2();
    options_.seed_converge_threshold = 1.0/200.0;
    options_.klt_epslion = 0.0001;
    options_.align_epslion = 0.0001;
    options_.min_disparity = 100;
    options_.min_track_features = 50;
    options_.min_redundant_observations = 3;
}

void LocalMapper::setThread(bool enable_main, bool enable_track)
{
    status_track_thread_ = enable_track;
    if(enable_main && mapping_thread_ == nullptr)
        mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
    else if(!enable_main && mapping_thread_ != nullptr)
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

void LocalMapper::drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst)
{
    //! draw features
    const cv::Mat src = frame->getImage(0);
    std::vector<Feature::Ptr> fts;
    frame->getFeatures(fts);
    cv::cvtColor(src, dst, CV_GRAY2RGB);
    int font_face = 1;
    double font_scale = 0.5;
    for(const Feature::Ptr &ft : fts)
    {
        Vector2d ft_px = ft->px;
        cv::Point2f px(ft_px[0], ft_px[1]);
        cv::Scalar color(0, 255, 0);
        cv::circle(dst, px, 2, color, -1);

        string id_str = std::to_string((frame->Tcw()*ft->mpt->pose()).norm());//ft->mpt->getFoundRatio());//
        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

    //! draw seeds
    for(const Seed::Ptr &seed : tracked_seeds_)
    {
        cv::Point2f px(seed->px_cur[0], seed->px_cur[1]);
        double convergence = seed->z_range/std::sqrt(seed->sigma2);
        double scale = MIN(convergence, 256.0) / 256.0;
        cv::Scalar color(255*scale, 0, 255*(1-scale));
        cv::circle(dst, px, 2, color, -1);

//        string id_str = std::to_string();
//        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = KeyFrame::create(frame_cur);

    //! before import, make sure the features are stored in the same order!
    std::vector<Feature::Ptr> fts_ref, fts_cur;
    keyframe_ref->getFeatures(fts_ref);
    keyframe_cur->getFeatures(fts_cur);

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    for(size_t i = 0; i < N; i++)
    {
        fts_ref[i]->mpt->addObservation(keyframe_ref, fts_ref[i]);
        fts_cur[i]->mpt->addObservation(keyframe_cur, fts_cur[i]);
    }

    for(Feature::Ptr ft : fts_ref)
    {
        map_->insertMapPoint(ft->mpt);
        ft->mpt->resetType(MapPoint::STABLE);
        ft->mpt->updateViewAndDepth();
    }

    current_frame_ = frame_cur;
    keyframe_ref->updateConnections();
    keyframe_cur->updateConnections();
    insertKeyFrame(keyframe_ref);
    insertKeyFrame(keyframe_cur);

    LOG_IF(INFO, report_) << "[Mapping] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

KeyFrame::Ptr LocalMapper::getReferenceKeyFrame()
{
    return current_keyframe_;
}

void LocalMapper::insertFrame(const Frame::Ptr &frame)
{
    LOG_ASSERT(frame != nullptr) << "[Mapping] Error input! Frame should not be null!";
    last_frame_ = current_frame_;
    current_frame_ = frame;
    if(status_track_thread_)
    {
        frame_process_future_ = std::async(std::launch::async, &LocalMapper::trackSeeds, this);
    }
    else
    {
        trackSeeds();
    }
}

bool LocalMapper::finishFrame()
{
    //! wait if trackSeeds() did not finish
    if(status_track_thread_)
    {
        frame_process_future_.wait();
        uint64_t id = frame_process_future_.get();
        LOG_ASSERT( id == current_frame_->id_) << "[Mapping] Wrong frame input! " << id << " != " << current_frame_->id_;
    }

    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_frame_);
        cond_process_.notify_one();
    }
    else
    {
        updateSeeds();
        reprojectSeeds();
        if(needCreateKeyFrame())
            processNewKeyFrame();
    }

//    std::vector<MapPoint::Ptr> mpts = map_->getAllMapPoints();
//    std::vector<int> histogram(10, 0);
//    std::vector<int> histogram_low(10, 0);
//    int size_low = 0;
//    for(const MapPoint::Ptr &mpt : mpts)
//    {
//        int hist = int (mpt->getFoundRatio() * 9.99);
//        histogram[hist]++;
//        if(hist == 0)
//        {
//            int hist_low = int (mpt->getFoundRatio() * 99.9);
//            histogram_low[hist_low]++;
//            size_low++;
//        }
//    }
//    std::string log = " [ ";
//    for(int i = 0; i < 10; ++i)
//    {
//        log += std::to_string((float)histogram[i] / mpts.size()) + " ";
//    }
//    log += "], " + std::to_string(mpts.size());
//    LOG(ERROR) << log;
//    log = " [ ";
//    for(int i = 0; i < 10; ++i)
//    {
//        log += std::to_string(histogram_low[i]) + " ";
//    }
//    log += "], " + std::to_string(size_low);
//    LOG(ERROR) << log;

    return true;
}

bool LocalMapper::needCreateKeyFrame()
{
    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();
    const int overlap = overlap_kfs[current_keyframe_];

    std::vector<Feature::Ptr> fts;
    current_frame_->getFeatures(fts);
    std::map<MapPoint::Ptr, Feature::Ptr> mpt_ft;
    for(const Feature::Ptr &ft : fts)
    {
        mpt_ft.emplace(ft->mpt, ft);
    }

    KeyFrame::Ptr max_overlap_keyframe;
    int max_overlap = 0;
    for(const auto &olp_kf : overlap_kfs)
    {
        if(olp_kf.second < max_overlap || (olp_kf.second == max_overlap && olp_kf.first->id_ < max_overlap_keyframe->id_))
            continue;

        max_overlap_keyframe = olp_kf.first;
        max_overlap = olp_kf.second;
    }

    //! check distance
    bool c1 = true;
    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);
    for(const auto &ovlp_kf : overlap_kfs)
    {
        SE3d T_cur_from_ref = current_frame_->Tcw() * ovlp_kf.first->pose();
        Vector3d tran = T_cur_from_ref.translation();
        double dist1 = tran.dot(tran);
        double dist2 = 0.2 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
        if(dist1 + dist2 < 0.12 * median_depth)
        {
            c1 = false;
            break;
        }
    }

    //! check disparity
    std::list<float> disparities;
    const int threahold = (int)max_overlap * 0.6;
    for(const auto &ovlp_kf : overlap_kfs)
    {
        if(ovlp_kf.second < threahold)
            continue;

        std::vector<float> disparity;
        disparity.reserve(ovlp_kf.second);
        MapPoints mpts;
        ovlp_kf.first->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            Feature::Ptr ft_ref = mpt->findObservation(ovlp_kf.first);
            Feature::Ptr ft_cur = mpt_ft[mpt];
            if(ft_ref != nullptr && ft_cur != nullptr)
            {
                const Vector2d px(ft_ref->px - ft_cur->px);
                disparity.push_back(px.norm());
            }
        }

        std::sort(disparity.begin(), disparity.end());
        float disp = disparity.at(disparity.size()/2);
        disparities.push_back(disp);
    }
    disparities.sort();

    LOG_IF(INFO, report_) << "[Mapping] Max overlap: " << max_overlap << " min disaprity " << disparities.front() << ", average ";

    bool c2 = disparities.front() > options_.min_disparity;
    bool c3 = current_frame_->N() < current_keyframe_->N() * 0.5;

    //! create new keyFrame
    if(c1 || c2 || c3)
    {
//        LOG(ERROR) << "C: (" << c1 << ", " << c2 << ", " << c3 << ") cur_n: " << current_frame_->N() << " ck: " << current_keyframe_->N();
        return true;
    }
    //! change reference keyframe
    else
    {
        if(overlap_kfs[current_keyframe_] < max_overlap * 0.85)
            current_keyframe_ = max_overlap_keyframe;
    }

    return false;
}

void LocalMapper::processNewKeyFrame()
{
    //! create new keyframe
    KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);
    std::vector<Feature::Ptr> fts;
    new_keyframe->getFeatures(fts);
    for(const Feature::Ptr &ft : fts)
    {
        ft->mpt->addObservation(new_keyframe, ft);
        ft->mpt->updateViewAndDepth();
    }
    new_keyframe->updateConnections();

    //! insert to mapper
    insertKeyFrame(new_keyframe);
}

void LocalMapper::insertKeyFrame(const KeyFrame::Ptr &keyframe)
{
    map_->insertKeyFrame(keyframe);
    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_keyframe_);
        current_keyframe_ = keyframe;
        cond_process_.notify_one();
    }
    else
    {
        current_keyframe_ = keyframe;

        std::list<MapPoint::Ptr> bad_mpts;
        int new_features = 0;
        if(map_->kfs_.size() > 2)
        {
            new_features = createFeatureFromLocalMap();
            Optimizer::localBundleAdjustment(current_keyframe_, bad_mpts, report_, verbose_);
        }

        for(const MapPoint::Ptr &mpt : bad_mpts)
        {
            map_->removeMapPoint(mpt);
        }

        int new_seeds = createSeeds(map_->kfs_.size() > 1);//! start from the second kf

        LOG_IF(INFO, report_) << "[Mapping] New created features:" << new_features << ", seeds: " << new_seeds;

//        checkCulling();
    }
}

bool LocalMapper::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_keyframe_);
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

    std::vector<Feature::Ptr> fts;
    current_keyframe_->getFeatures(fts);

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    // TODO 如果对应的seed收敛，在跟踪过的关键帧增加观测？
    if(current_frame_ != nullptr && current_keyframe_->frame_id_ == current_frame_->id_)
    {
        for(const Seed::Ptr &seed : tracked_seeds_)
        {
            const Vector2d &px = seed->px_cur;
            old_corners.emplace_back(Corner(px[0], px[1], 0, seed->level_cur));
        }
    }

    Corners new_corners;
    fast_detector_->detect(current_keyframe_->images(), new_corners, old_corners, 150);

    if(new_corners.empty())
        return 0;

    double depth_mean;
    double depth_min;
    current_keyframe_->getSceneDepth(depth_mean, depth_min);
    Seeds new_seeds;
    for(const Corner &corner : new_corners)
    {
        const Vector2d px(corner.x, corner.y);
        const Vector3d fn(current_keyframe_->cam_->lift(px));
        new_seeds.emplace_back(Seed::create(current_keyframe_, px, fn, 0, depth_mean, depth_min));
    }
    seeds_buffer_.emplace_back(current_keyframe_, std::make_shared<Seeds>(new_seeds));

    if(is_track)
    {
        for(const Seed::Ptr &seed : new_seeds)
            tracked_seeds_.emplace_back(seed);
    }

    return (int)new_seeds.size();
}

int LocalMapper::createFeatureFromLocalMap()
{
    double t0 = (double)cv::getTickCount();
    std::set<KeyFrame::Ptr> connected_keyframes = current_keyframe_->getConnectedKeyFrames();
    std::set<KeyFrame::Ptr> local_keyframes = connected_keyframes;

    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            local_keyframes.insert(sub_kf);
        }
    }


    double t1 = (double)cv::getTickCount();
    std::unordered_set<MapPoint::Ptr> local_mpts;
    MapPoints mpts_cur;
    current_keyframe_->getMapPoints(mpts_cur);
    for(const MapPoint::Ptr &mpt : mpts_cur)
    {
        local_mpts.insert(mpt);
    }

    std::unordered_set<MapPoint::Ptr> candidate_mpts;
    for(const KeyFrame::Ptr &kf : local_keyframes)
    {
        MapPoints mpts;
        kf->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            if(local_mpts.count(mpt) || candidate_mpts.count(mpt))
                continue;

            if(mpt->isBad())//! however it should not happen, maybe still some bugs in somewhere
            {
                kf->removeMapPoint(mpt);
                continue;
            }

//            if(mpt->getFoundRatio() < 0.1)
//            {
//                mpt->setBad();
//                map_->removeMapPoint(mpt);
//                continue;
//            }

            candidate_mpts.insert(mpt);
        }
    }
    double t2 = (double)cv::getTickCount();

    //! match the mappoints from nearby keyframes
    int project_count = 0;
    std::list<Feature::Ptr> new_fts;
    for(const MapPoint::Ptr &mpt : candidate_mpts)
    {
        Vector3d xyz_cur(current_keyframe_->Tcw() * mpt->pose());
        if(xyz_cur[2] < 0.0f)
            continue;

        Vector2d px_cur(current_keyframe_->cam_->project(xyz_cur));
        if(!current_keyframe_->cam_->isInFrame(px_cur.cast<int>(), 8))
            continue;

        project_count++;

        int level_cur = 0;
        int result = FeatureTracker::reprojectMapPoint(current_keyframe_, mpt, px_cur, level_cur, 15, 0.01);
        if(result != 1)
            continue;

        Vector3d ft_cur = current_keyframe_->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        new_fts.push_back(new_feature);
    }

    //! check whether the matched corner is near a exsit corner.
    //! firstly, create a mask for exsit corners
    const int cols = current_keyframe_->cam_->width();
    const int rows = current_keyframe_->cam_->height();
    cv::Mat mask(rows, cols, CV_16SC1, -1);
    std::vector<Feature::Ptr> old_fts;
    current_keyframe_->getFeatures(old_fts);
    const int old_fts_size = (int) old_fts.size();
    for(int i = 0; i < old_fts_size; ++i)
    {
        const Vector2i px = old_fts[i]->px.cast<int>();
        for(int c = -1; c <= 1; ++c)
        {
            int16_t* ptr = mask.ptr<int16_t>(px[1]+c) + px[0];
            ptr[-1] = (int16_t)i;
            ptr[0] = (int16_t)i;
            ptr[1] = (int16_t)i;
        }
    }

    //! check whether the mappoint is already exist
    int created_count = 0;
    int fusion_count = 0;
    for(const Feature::Ptr &ft : new_fts)
    {
        const Vector2i px = ft->px.cast<int>();
        int64_t id = mask.ptr<int16_t>(px[1])[px[0]];
        //! if not occupied, create new feature
        if(id == -1)
        {
            //! create new features
            current_keyframe_->addFeature(ft);
            ft->mpt->addObservation(current_keyframe_, ft);
            ft->mpt->increaseVisible(2);
            ft->mpt->increaseFound(2);
            created_count++;
            LOG_IF(INFO, verbose_) << " create new feature from mpt " << ft->mpt->id_;
        }
        //! if already occupied, check whether the mappoint is the same
        else
        {
            MapPoint::Ptr mpt_new = ft->mpt;
            MapPoint::Ptr mpt_old = old_fts[id]->mpt;
            const std::map<KeyFrame::Ptr, Feature::Ptr> obs_new = mpt_new->getObservations();
            const std::map<KeyFrame::Ptr, Feature::Ptr> obs_old = mpt_old->getObservations();

            bool is_same = true;
            std::list<double> squared_dist;
            std::unordered_set<KeyFrame::Ptr> sharing_keyframes;
            for(const auto &it : obs_new)
            {
                const auto it_old = obs_old.find(it.first);
                if(it_old == obs_old.end())
                    continue;

                const Feature::Ptr &ft_old = it_old->second;
                const Feature::Ptr &ft_new = it.second;
                Vector2d px_delta(ft_new->px - ft_old->px);
                squared_dist.push_back(px_delta.squaredNorm());
                is_same &= squared_dist.back() < 1.0; //! only if all the points pair match the conditon

                sharing_keyframes.insert(it.first);
            }

            if(!squared_dist.empty() && !is_same)
            {
//                std::cout << " ***=-=*** ";
//                std::for_each(squared_dist.begin(), squared_dist.end(), [](double dis){std::cout << dis << ", ";});
//                std::cout << std::endl;
//                goto SHOW;
                continue;
            }


            if(obs_old.size() >= obs_new.size())
            {
                //! match all ft in obs_new
                std::list<std::tuple<Feature::Ptr, double, double, int> > fts_to_update;
                for(const auto &it_new : obs_new)
                {
                    const KeyFrame::Ptr &kf_new = it_new.first;
                    if(sharing_keyframes.count(kf_new))
                        continue;

                    const Vector3d kf_new_dir(kf_new->ray().normalized());
                    double max_cos_angle = 0;
                    KeyFrame::Ptr kf_old_ref;
                    for(const auto &it_old : obs_old)
                    {
                        Vector3d kf_old_dir(it_old.first->ray().normalized());
                        double view_cos_angle = kf_old_dir.dot(kf_new_dir);

                        //! find min angle, max cosangle
                        if(view_cos_angle < max_cos_angle)
                            continue;

                        max_cos_angle = view_cos_angle;
                        kf_old_ref = it_old.first;
                    }

                    Feature::Ptr ft_old = obs_old.find(kf_old_ref)->second;
                    Vector3d xyz_new(kf_new->Tcw() * ft_old->mpt->pose());
                    if(xyz_new[2] < 0.0f)
                        continue;

                    Vector2d px_new(kf_new->cam_->project(xyz_new));
                    if(!kf_new->cam_->isInFrame(px_new.cast<int>(), 8))
                        continue;

                    int level_new = 0;
                    bool matched = FeatureTracker::trackFeature(kf_old_ref, kf_new, ft_old, px_new, level_new, 15, 0.01, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_new.find(kf_new)->second, px_new[0], px_new[1], level_new);
                }

                //! update ft if succeed
                const Camera::Ptr &cam = current_keyframe_->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px[0] = std::get<1>(it);
                    ft_update->px[1] = std::get<2>(it);
                    ft_update->level = std::get<3>(it);
                    ft_update->fn = cam->lift(ft_update->px);
                }

                //! fusion the mappoint
                //! just reject the new one
                mpt_old->fusion(mpt_new);
                map_->removeMapPoint(mpt_new);

                LOG_IF(INFO, verbose_) << " Fusion mpt " << mpt_old->id_ << " with mpt " << mpt_new->id_;
//                goto SHOW;
            }
            else
            {
                //! match all ft in obs_old
                std::list<std::tuple<Feature::Ptr, double, double, int> > fts_to_update;
                for(const auto &it_old : obs_old)
                {
                    const KeyFrame::Ptr &kf_old = it_old.first;
                    if(sharing_keyframes.count(kf_old))
                        continue;

                    const Vector3d kf_old_dir(kf_old->ray().normalized());
                    double max_cos_angle = 0;
                    KeyFrame::Ptr kf_new_ref;
                    for(const auto &it_new : obs_new)
                    {
                        Vector3d kf_new_dir(it_new.first->ray().normalized());
                        double view_cos_angle = kf_new_dir.dot(kf_old_dir);

                        //! find min angle, max cosangle
                        if(view_cos_angle < max_cos_angle)
                            continue;

                        max_cos_angle = view_cos_angle;
                        kf_new_ref = it_new.first;
                    }

                    Feature::Ptr ft_new = obs_new.find(kf_new_ref)->second;

                    Vector3d xyz_old(kf_old->Tcw() * ft_new->mpt->pose());
                    if(xyz_old[2] < 0.0f)
                        continue;

                    Vector2d px_old(kf_old->cam_->project(xyz_old));
                    if(!kf_old->cam_->isInFrame(px_old.cast<int>(), 8))
                        continue;

                    int level_old = 0;
                    bool matched = FeatureTracker::trackFeature(kf_new_ref, kf_old, ft_new, px_old, level_old, 15, 0.01, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_old.find(kf_old)->second, px_old[0], px_old[1], level_old);
                }

                //! update ft if succeed
                const Camera::Ptr &cam = current_keyframe_->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px[0] = std::get<1>(it);
                    ft_update->px[1] = std::get<2>(it);
                    ft_update->level = std::get<3>(it);
                    ft_update->fn = cam->lift(ft_update->px);
                }

                //! add new feature for keyframe, then fusion the mappoint
                ft->mpt = mpt_new;
                current_keyframe_->addFeature(ft);
                mpt_new->addObservation(current_keyframe_, ft);

                mpt_new->fusion(mpt_old);
                map_->removeMapPoint(mpt_old);

                LOG_IF(INFO, verbose_) << " Fusion mpt " << mpt_new->id_ << " with mpt " << mpt_old->id_;
//                goto SHOW;
            }

            fusion_count++;
            continue;

//            SHOW:
//            std::cout << " mpt_new: " << mpt_new->id_ << ", " << mpt_new->pose().transpose() << std::endl;
//            for(const auto &it : obs_new)
//            {
//                std::cout << "-kf: " << it.first->id_ << " px: [" << it.second->px[0] << ", " << it.second->px[1] << "]" << std::endl;
//            }
//
//            std::cout << " mpt_old: " << mpt_old->id_ << ", " << mpt_old->pose().transpose() << std::endl;
//            for(const auto &it : obs_old)
//            {
//                std::cout << "=kf: " << it.first->id_ << " px: [" << it.second->px[0] << ", " << it.second->px[1] << "]" << std::endl;
//            }
//
//            for(const auto &it : obs_new)
//            {
//                string name = "new -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px[0]/(1<<it.second->level), it.second->px[1]/(1<<it.second->level));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//
//            for(const auto &it : obs_old)
//            {
//                string name = "old -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px[0]/(1<<it.second->level), it.second->px[1]/(1<<it.second->level));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//            cv::waitKey(0);
        }

    }

    double t3 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[Mapping][*] New Featrue Time: "
                             << (t1-t0)/cv::getTickFrequency() << " "
                             << (t2-t1)/cv::getTickFrequency() << " "
                             << (t3-t2)/cv::getTickFrequency() << " "
                             << " old points: " << mpts_cur.size() << " new projected: " << project_count << "(" << candidate_mpts.size() << ")"
                             << ", points matched: " << new_fts.size() << " with " << created_count << " created, " << fusion_count << " fusioned. ";

    return 0;
}


uint64_t LocalMapper::trackSeeds()
{
    if(current_frame_ == nullptr || last_frame_ == nullptr)
        return 0;

    //! track seeds by klt
    std::vector<cv::Point2f> pts_to_track;
    pts_to_track.reserve(tracked_seeds_.size());
    for(const Seed::Ptr &seed : tracked_seeds_)
    {
        pts_to_track.emplace_back(cv::Point2f(seed->px_cur[0], seed->px_cur[1]));
    }

    if(pts_to_track.empty())
        return current_frame_->id_;

    std::vector<cv::Point2f> pts_tracked = pts_to_track;
    std::vector<bool> status;
    static cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, options_.klt_epslion);
    utils::kltTrack(last_frame_->opticalImages(), current_frame_->opticalImages(), Frame::optical_win_size_,
                    pts_to_track, pts_tracked, status, termcrit, true, verbose_);

    //! erase untracked seeds
    size_t idx = 0;
    int count = 0;
    std::vector<float> disparity;
    disparity.reserve(pts_to_track.size());
    auto track_seeds_itr = tracked_seeds_.begin();
    for(;track_seeds_itr!=tracked_seeds_.end(); idx++)
    {
        if(status[idx])
        {
            const cv::Point2f &px = pts_tracked[idx];
            Vector2d &px_cur = (*track_seeds_itr)->px_cur;
            px_cur[0] = px.x;
            px_cur[1] = px.y;
            const cv::Point2f disp = pts_to_track[idx] - pts_tracked[idx];
            const float dist = disp.x*disp.x+disp.y+disp.y;
            disparity .push_back(dist);
            track_seeds_itr++;
            count++;
        }
        else
        {
            track_seeds_itr = tracked_seeds_.erase(track_seeds_itr);
        }
    }

    LOG_IF(INFO, report_) << "[Mapping][1] Tracking seeds: " << count;

    return current_frame_->id_;
}

int LocalMapper::updateSeeds()
{
    //! remove error tracked seeds and update
    std::map<KeyFrame::Ptr, std::deque<Seed::Ptr> > seeds_map;
    for(const Seed::Ptr &track_seed : tracked_seeds_)
    {
        const auto seeds_map_itr = seeds_map.lower_bound(track_seed->kf);
        if(seeds_map_itr == seeds_map.end())
        {
            std::deque<Seed::Ptr> new_deque;
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
        const std::deque<Seed::Ptr> &seeds_deque = seed_map.second;
        const SE3d T_cur_from_ref = current_frame_->Tcw() * kf->pose();
        const SE3d T_ref_from_cur = T_cur_from_ref.inverse();
        for(const Seed::Ptr &seed : seeds_deque)
        {
            const Vector3d fn_cur = current_frame_->cam_->lift(seed->px_cur);
            double err2 = utils::Fundamental::computeErrorSquared(
                kf->pose().translation(), seed->fn_ref/seed->mu, T_cur_from_ref, fn_cur.head<2>());

            if(err2 > options_.epl_dist2_threshold)
                continue;

            //! update
            double depth = -1;
            if(utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), seed->fn_ref, fn_cur, depth))
            {
                double tau = seed->computeTau(T_ref_from_cur, seed->fn_ref, depth, px_error_angle);
//                tau = seed->computeVar(T_cur_from_ref, depth, options_.klt_epslion);
//                tau = tau + Config::pixelUnSigma();
                seed->update(1.0/depth, tau*tau);

                //! check converge
                if(seed->sigma2 / seed->z_range < options_.seed_converge_threshold)
                {
                    createFeatureFromSeed(seed);
                    earseSeed(seed->kf, seed);
                    continue;
                }
            }

            tracked_seeds_.emplace_back(seed);
        }
    }

    LOG_IF(INFO, report_) << "[Mapping][2] Seeds updated: " << tracked_seeds_.size();
    return (int)tracked_seeds_.size();
}

int LocalMapper::reprojectSeeds()
{
    //! get new seeds for track
    std::set<KeyFrame::Ptr> candidate_keyframes = current_frame_->getRefKeyFrame()->getConnectedKeyFrames(options_.max_kfs);
    candidate_keyframes.insert(current_frame_->getRefKeyFrame());

    std::set<Seed::Ptr> seed_tracking;
    for(const Seed::Ptr &seed : tracked_seeds_)
        seed_tracking.insert(seed);

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
            double dist2 = utils::Fundamental::computeErrorSquared(keyframe->pose().translation(), seed->fn_ref/seed->mu, T_cur_from_ref, fn_matched);
            if(dist2 > options_.epl_dist2_threshold)
            {
                seed_itr++;
                continue;
            }

            double depth = -1;
            const Vector3d fn_cur = current_frame_->cam_->lift(px_matched);
            bool succeed = utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), seed->fn_ref, fn_cur, depth);
            if(!succeed)
            {
                seed_itr++;
                continue;
            }

            double tau = seed->computeTau(T_ref_from_cur, seed->fn_ref, depth, px_error_angle);
//            tau = seed->computeVar(T_cur_from_ref, depth, options_.align_epslion);
//            tau = tau + Config::pixelUnSigma();
            seed->update(1.0/depth, tau*tau);

            //! check converge
            if(seed->sigma2 / seed->z_range < options_.seed_converge_threshold)
            {
                createFeatureFromSeed(seed);
                seed_itr = seeds.erase(seed_itr);
                continue;
            }

            //! update px
            seed->px_cur = px_matched;
            tracked_seeds_.emplace_back(seed);
            count++;
            seed_itr++;
        }

        if(seeds.empty())
            buffer_itr = seeds_buffer_.erase(buffer_itr);
        else
            buffer_itr++;
    }

    LOG_IF(INFO, report_) << "[Mapping][3] Seeds after reprojected: " << tracked_seeds_.size() << "(" << count << ")";

    return (int) tracked_seeds_.size();
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

bool LocalMapper::createFeatureFromSeed(const Seed::Ptr &seed)
{
    //! create new feature
    // TODO add_observation 不用，需不需要找一下其他关键帧是否观测改点，增加约束
    // TODO 把这部分放入一个队列，在单独线程进行处理
    MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->mu));
    Feature::Ptr ft = Feature::create(seed->px_ref, seed->fn_ref, seed->level_ref, mpt);
    seed->kf->addFeature(ft);
    map_->insertMapPoint(mpt);
    mpt->addObservation(seed->kf, ft);
    mpt->updateViewAndDepth();
//    std::cout << " Create new seed as mpt: " << ft->mpt->id_ << ", " << 1.0/seed->mu << ", kf: " << seed->kf->id_ << " his: ";
//    for(const auto his : seed->history){ std::cout << "[" << his.first << "," << his.second << "]";}
//    std::cout << std::endl;

    return true;
}

void LocalMapper::checkCulling()
{
    const std::set<KeyFrame::Ptr> connected_keyframes = current_keyframe_->getConnectedKeyFrames();

    int count = 0;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        if(kf->id_ == 0)
            continue;

        const int observations_threshold = 3;
        int observations = 0;
        int redundant_observations = 0;
        MapPoints mpts;
        kf->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
            if(obs.size() > observations_threshold)
            {
                const Feature::Ptr &ft = obs[kf];
                for(const auto &it : obs)
                {
                    if(it.first == kf)
                        continue;

                    if(it.second->level <= ft->level+1)
                    {
                        observations++;
                        if(observations >= options_.min_redundant_observations)
                            break;
                    }
                }

                if(observations >= options_.min_redundant_observations)
                    redundant_observations++;
            }

            if(redundant_observations > mpts.size() * 0.9)
            {
                kf->setBad();
                map_->removeKeyFrame(kf);
                count++;
            }
        }

    }
}

bool LocalMapper::findEpipolarMatch(const Seed::Ptr &seed,
                                    const KeyFrame::Ptr &keyframe,
                                    const Frame::Ptr &frame,
                                    const SE3d &T_cur_from_ref,
                                    Vector2d &px_matched,
                                    int &level_matched)
{
    static const int patch_size = AlignPatch::Size;
//    static const int patch_area = AlignPatch::Area;
    static const int half_patch_size = AlignPatch::HalfSize;

    //! check if in the view of current frame
    const double z_ref = 1.0/seed->mu;
    const Vector3d xyz_ref(seed->fn_ref * z_ref);
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
    Vector3d xyz_near = T_cur_from_ref * (seed->fn_ref * z_ref_min);
    Vector3d xyz_far  = T_cur_from_ref * (seed->fn_ref * z_ref_max);
    xyz_near /= xyz_near[2];
    xyz_far  /= xyz_far[2];
    Vector2d epl_dir = (xyz_near - xyz_far).head<2>();

    //! calculate best search level
    const int level_ref = seed->level_ref;
    const int level_cur = MapPoint::predictScale(z_ref, z_cur, level_ref, frame->max_level_);
    seed->level_cur = level_cur;
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

    utils::getWarpMatrixAffine(keyframe->cam_, frame->cam_, seed->px_ref, seed->fn_ref, level_ref,
                               z_ref, T_cur_from_ref, patch_size, A_cur_from_ref);

//    double det = A_cur_from_ref.determinant() / factor;
//    std::cout << "***** det: " <<  det << std::endl;

    static const int patch_border_size = patch_size+2;
    cv::Mat image_ref = keyframe->getImage(level_ref);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                 seed->px_ref, level_ref, level_cur);

    Matrix<float, patch_size, patch_size, RowMajor> patch;
    patch = patch_with_border.block(1, 1, patch_size, patch_size);

    const cv::Mat image_cur = frame->getImage(level_cur);

    Vector2d px_best(-1,-1);
    if(epl_length > 2.0)
    {
        int n_steps = epl_length / 0.707;
        Vector2d step = epl_dir / n_steps;

        // TODO 使用模板来加速！！！
        //! SSD
        double t0 = (double)cv::getTickCount();
        ZSSD<float, patch_size> zssd(patch);
        double score_best = std::numeric_limits<double>::max();
        double score_second = score_best;
        int index_best = -1;
        int index_second = -1;

        Vector2d fn_start = xyz_far.head<2>() - step * 2;
        n_steps += 2;
        Vector2d fn(fn_start);
        for(int i = 0; i < n_steps; ++i, fn += step)
        {
            Vector2d px(frame->cam_->project(fn[0], fn[1]) / factor);

            //! always in frame's view
            if(!frame->cam_->isInFrame(px.cast<int>(), half_patch_size, level_cur))
                continue;

            Matrix<float, patch_size, patch_size, RowMajor> patch_cur;
            utils::interpolateMat<uchar, float, patch_size>(image_cur, patch_cur, px[0], px[1]);

            float score = zssd.compute_score(patch_cur);

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
        px_best = frame->cam_->project(pn_best[0], pn_best[1]) / factor;

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

    Vector3d estimate(0,0,0); estimate.head<2>() = px_best;
    if(!AlignPatch::align2DI(image_cur, patch_with_border, estimate, 30, options_.align_epslion, verbose_))
    {
//        std::cout << "dx:\n " << dx << std::endl;
//        std::cout << "dy:\n " << dy << std::endl;

        static bool show = false;
        if(show)
        {
            //        showMatch(keyframe->getImage(level_ref), current_frame_->getImage(level_cur), px_near, px_far, ft->px/factor, px_best);
            showEplMatch(keyframe, frame, T_cur_from_ref, level_ref, level_cur, xyz_near, xyz_far, xyz_ref, px_best);
            showAffine(keyframe->getImage(level_ref), seed->px_ref/factor, A_cur_from_ref.inverse(), 8, level_ref);
        }

        return false;
    }

    //! transform to level-0
    px_matched = estimate.head<2>() * factor;

    LOG_IF(INFO, verbose_) << "Found! [" << seed->px_ref.transpose() << "] "
                           << "dst: [" << px_matched.transpose() << "] "
                           << "epl: [" << px_near.transpose() << "]--[" << px_far.transpose() << "]" << std::endl;

//    showMatch(image_ref, image_cur, px_near, px_far, seed.ft->px/factor, px_matched/factor);

    return true;
}


}