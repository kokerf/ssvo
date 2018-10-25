#include <future>
#include "config.hpp"
#include "utils.hpp"
#include "depth_filter.hpp"
#include "feature_alignment.hpp"
#include "image_alignment.hpp"
#include "time_tracing.hpp"
#include "feature_tracker.hpp"

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

    Vector3d epl_near = xyz_near / xyz_near[2];
    Vector3d epl_far  = xyz_far / xyz_far[2];

    Vector2d epl0 = frame->cam_->project(epl_near);
    Vector2d epl1 = frame->cam_->project(epl_far);

    const double scale_cur = Frame::inv_scale_factors_.at(level_cur);
    const double scale_ref = Frame::inv_scale_factors_.at(level_ref);
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

//! =================================================================================================
TimeTracing::Ptr dfltTrace = nullptr;

//! DepthFilter
DepthFilter::DepthFilter(const FastDetector::Ptr &fast_detector, bool report, bool verbose) :
    fast_detector_(fast_detector), report_(report), verbose_(report&&verbose), filter_thread_(nullptr), stop_require_(false)
{
    options_.max_kfs = 5;
    options_.max_seeds_buffer = 2;
    options_.max_features = Config::minCornersPerKeyFrame();
    options_.max_epl_length = 1000;
    options_.epl_dist2_threshold = 16;
    options_.klt_epslion = 0.0001;
    options_.align_epslion = 0.0001;
    options_.max_perprocess_kfs = Config::maxPerprocessKeyFrames();
    options_.pixel_error_threshold = 1;
    options_.min_frame_disparity = 0.0;//2.0;
    options_.min_pixel_disparity = 4.5;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("epl_search");
    time_names.push_back("create_seeds");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("num_new_seeds");
    log_names.push_back("num_epl_search");

    string trace_dir = Config::timeTracingDirectory();
    dfltTrace.reset(new TimeTracing("ssvo_trace_filter", trace_dir, time_names, log_names));
}

void DepthFilter::setSeedConvergedCallback(const SeedCallback &callback)
{
    seed_converged_callback_ = callback;
}

void DepthFilter::setKeyFrameProcessCallback(const KeyFrameCallback &callback)
{
    keyframe_process_callback_ = callback;
}

void DepthFilter::setKeyFrameSeedsCallback(const KeyFrameCallback &callback)
{
    keyframe_seeds_callback_ = callback;
}

void DepthFilter::startMainThread()
{
    if(filter_thread_ == nullptr)
        filter_thread_ = std::make_shared<std::thread>(std::bind(&DepthFilter::run, this));
}

void DepthFilter::stopMainThread()
{
    setStop();
    if(filter_thread_)
    {
        if(filter_thread_->joinable())
            filter_thread_->join();
        filter_thread_.reset();
    }
}

void DepthFilter::setStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    stop_require_ = true;
}

bool DepthFilter::isRequiredStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    return stop_require_;
}

void DepthFilter::run()
{
    LOG(WARNING) << "[Filter][*] Start main thread!";
    while(!isRequiredStop())
    {
        Frame::Ptr frame;
        KeyFrame::Ptr keyframe;
        if(checkNewFrame(frame, keyframe))
        {
            dfltTrace->startTimer("total");

            int searched_count = 0;
            int new_seeds_count = 0;

            dfltTrace->startTimer("epl_search");
            searched_count = reprojectAllSeeds(frame);
            dfltTrace->stopTimer("epl_search");
            dfltTrace->log("num_epl_search", searched_count);

            if(keyframe)
            {
                dfltTrace->startTimer("create_seeds");
                new_seeds_count = createSeeds(keyframe);
                dfltTrace->stopTimer("create_seeds");
                dfltTrace->log("num_new_seeds", new_seeds_count);

                seeds_buffer_.push_back(keyframe);
                if(seeds_buffer_.size() > options_.max_seeds_buffer)
                    seeds_buffer_.pop_front();

                keyframe_process_callback_(keyframe);
//                updateByConnectedKeyFrames(keyframe, 3);
            }

            dfltTrace->stopTimer("total");

            const double time_total = dfltTrace->getTimer("total");
            const double time_epl_search = dfltTrace->getTimer("epl_search");
            const double time_seeds_create = dfltTrace->getTimer("create_seeds");
            LOG_IF(WARNING, report_) << "----------\n"
                                     << "[Filter][*] Frame: " << frame->id_ << "\n"
                                     << std::fixed << std::setprecision(6)
                                     << "> Tim(ms): " << time_total << " = " << time_epl_search << "(S) + " << time_seeds_create << "(C) + ...\n"
                                     << "> Seeds  : " << searched_count << "(S), " << new_seeds_count << "(C)";

            dfltTrace->writeToFile();
        }

    }
}

//void DepthFilter::logSeedsInfo()
//{
//    std::unique_lock<std::mutex> lock(mutex_seeds_);
//    std::list<std::tuple<uint64_t, int, int>> seeds_info;
//    for(const auto it : seeds_buffer_)
//    {
//        auto rate_itr = seeds_convergence_rate_.find(it.first->id_);
//        if(rate_itr!=seeds_convergence_rate_.end())
//            std::get<1>(rate_itr->second) = it.second->size();
//    }
//
//    for(const auto it : seeds_convergence_rate_)
//    {
//        seeds_info.emplace_back(it.first, std::get<0>(it.second), std::get<1>(it.second));
//    }
//
//    seeds_info.sort([](const std::tuple<uint64_t, int, int> &a, const std::tuple<uint64_t, int, int> &b){
//      return std::get<0>(a) < std::get<0>(b);
//    });
//
//    std::ofstream f;
//    f.open("/tmp/ssvo_seeds_rate.txt");
//
//    for(const auto &it : seeds_info)
//    {
//        f << std::get<0>(it) << " " << std::get<1>(it) << " " << std::get<2>(it) << " " << 1.0*std::get<2>(it)/std::get<1>(it)<< "\n";
//    }
//    f.flush();
//
//    f.close();
//}

bool DepthFilter::checkNewFrame(Frame::Ptr &frame, KeyFrame::Ptr &keyframe)
{
    std::unique_lock<std::mutex> lock(mutex_frame_);
    cond_process_main_.wait_for(lock, std::chrono::microseconds(5));

    if(frames_buffer_.empty())
        return false;

    frame = frames_buffer_.front().first;
    keyframe = frames_buffer_.front().second;
    frames_buffer_.pop_front();

    return frame != nullptr;
}

void DepthFilter::insertFrame(const Frame::Ptr &frame, const KeyFrame::Ptr keyframe)
{
    LOG_ASSERT(frame != nullptr) << "[Filter] Error input! Frame should not be null!";

    if(filter_thread_ == nullptr)
    {
        dfltTrace->startTimer("total");

        int searched_count = 0;
        int new_seeds_count = 0;

        dfltTrace->startTimer("epl_search");
        searched_count = reprojectAllSeeds(frame);
        dfltTrace->stopTimer("epl_search");
        dfltTrace->log("num_epl_search", searched_count);

        if(keyframe)
        {
            dfltTrace->startTimer("create_seeds");
            new_seeds_count = createSeeds(keyframe);
            dfltTrace->stopTimer("create_seeds");
            dfltTrace->log("num_new_seeds", new_seeds_count);

            seeds_buffer_.push_back(keyframe);
            if(seeds_buffer_.size() > options_.max_seeds_buffer)
                seeds_buffer_.pop_front();

            keyframe_process_callback_(keyframe);
//            updateByConnectedKeyFrames(keyframe, 3);
        }

        dfltTrace->stopTimer("total");

        const double time_total = dfltTrace->getTimer("total");
        const double time_epl_search = dfltTrace->getTimer("epl_search");
        const double time_seeds_create = dfltTrace->getTimer("create_seeds");
        LOG_IF(WARNING, report_) << "----------\n"
                                 << "[Filter][*] Frame: " << frame->id_ << "\n"
                                 << std::fixed << std::setprecision(6)
                                 << "> Tim(ms): " << time_total << " = " << time_epl_search << "(S) + " << time_seeds_create << "(C) + ...\n"
                                 << "> Seeds  : " << searched_count << "(S), " << new_seeds_count << "(C)";

        dfltTrace->writeToFile();

    }
    else
    {
        std::unique_lock<std::mutex> lock(mutex_frame_);
        frames_buffer_.emplace_back(frame, keyframe);
        cond_process_main_.notify_one();
    }
}

//int DepthFilter::getSeedsForMapping(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame)
//{
//    LOG_ASSERT(keyframe) << "KeyFrame is null!";
//    LOG_ASSERT(frame) << "Frame is null!";
//
//    updateSeeds(frame);
//
//    std::vector<Feature::Ptr> seeds;
//    frame->getSeeds(seeds);
//    for(const Feature::Ptr & ft_seed : seeds)
//    {
//        const Seed::Ptr &seed = ft_seed->seed_;
//        keyframe->addSeed(ft_seed);
//        frame->removeSeed(seed);
//        earseSeed(seed->kf, seed);
//    }
//
//    return (int) seeds.size();
//}

void DepthFilter::insertKeyFrame(const KeyFrame::Ptr &keyframe)
{
    createSeeds(keyframe);
    keyframe_process_callback_(keyframe);
}

int DepthFilter::createSeeds(const KeyFrame::Ptr &keyframe)
{
    if(keyframe == nullptr)
        return 0;

    keyframe->detectFast(fast_detector_);

    keyframe_seeds_callback_(keyframe);

    //return 0;
    
    std::vector<size_t> mpt_indices = keyframe->getMapPointMatchIndices();
    std::vector<size_t> seed_indices = keyframe->getSeedMatchIndices();

    std::unordered_set<size_t> mask_indices;
    for(const size_t &idx : mpt_indices)
        mask_indices.insert(idx);
    for(const size_t &idx : seed_indices)
        mask_indices.insert(idx);

    //! get immature features
    std::map<size_t, float> grid[KeyFrame::GRID_ROWS][KeyFrame::GRID_COLS];
    for(size_t r = 0; r < KeyFrame::GRID_ROWS; r++)
        for(size_t c = 0; c < KeyFrame::GRID_COLS; c++)
        {
            std::vector<size_t> fts_in_cell = keyframe->getFeaturesInGrid(r, c);
            for(const size_t& idx : fts_in_cell)
            {
                if(mask_indices.count(idx))
                {
                    grid[r][c].clear();
                    break;
                }

                const Feature::Ptr& ft = keyframe->getFeatureByIndex(idx);
                if(ft->corner_.score > 0)
                    grid[r][c].emplace(idx, ft->corner_.score);
            }
        }

    //! get grid size
    const size_t max_immature_fts = 200;
    const size_t max_step = std::sqrt(static_cast<float>(KeyFrame::GRID_ROWS*KeyFrame::GRID_COLS)/max_immature_fts);
    size_t step = max_step;
    for(;step > 1; step--)
    {
        size_t immature_fts = 0;
        for(size_t r = 0; r < KeyFrame::GRID_ROWS; r += step)
        {
            for(size_t c = 0; c < KeyFrame::GRID_COLS; c += step)
            {
                bool empty = true;
                const size_t max_rr = KeyFrame::GRID_ROWS < r + step ? KeyFrame::GRID_ROWS : r + step;
                for(size_t rr = r; rr < max_rr; rr++)
                {
                    const size_t max_cc = KeyFrame::GRID_COLS < c + step ? KeyFrame::GRID_COLS : c + step;
                    for(size_t cc = c; cc < max_cc; cc++)
                    {
                        if(!grid[rr][cc].empty())
                        {
                            empty = false;
                            break;
                        }
                    }

                    if(!empty)
                        break;
                }

                if(!empty)
                    immature_fts++;
            }
        }

        if(immature_fts >= max_immature_fts)
            break;
    }

    //! get seeds index
    std::list<size_t> mew_seed_indices;
    for(size_t r = 0; r < KeyFrame::GRID_ROWS; r += step)
    {
        for(size_t c = 0; c < KeyFrame::GRID_COLS; c += step)
        {
            size_t best_idx = -1;
            float best_score = -1;
            const size_t max_rr = KeyFrame::GRID_ROWS < r + step ? KeyFrame::GRID_ROWS : r + step;
            for(size_t rr = r; rr < max_rr; rr++)
            {
                const size_t max_cc = KeyFrame::GRID_COLS < c + step ? KeyFrame::GRID_COLS : c + step;
                for(size_t cc = c; cc < max_cc; cc++)
                {
                    for(const auto &it : grid[rr][cc])
                    {
                        const size_t& idx = it.first;
                        const float& score = it.second;

                        if(score > best_score)
                        {
                            best_idx = idx;
                            best_score = score;
                        }
                    }
                }
            }

            if(best_idx != -1)
                mew_seed_indices.push_back(best_idx);
        }
    }

    //FeatureTracker::showAllFeatures(keyframe);

    double depth_mean;
    double depth_min;
    keyframe->getSceneDepth(depth_mean, depth_min);
    for(const size_t &idx : mew_seed_indices)
    {
        const Seed::Ptr new_seed = Seed::create(keyframe, idx, depth_mean, depth_min);
        keyframe->addSeedFeatureCreated(new_seed, idx);
    }

    //FeatureTracker::showAllFeatures(keyframe);

    return (int)mew_seed_indices.size();
}

int DepthFilter::updateByConnectedKeyFrames(const KeyFrame::Ptr &keyframe, int num)
{
    const double focus_length = MIN(keyframe->cam_->fx(), keyframe->cam_->fy());
    const double pixel_usigma = Config::imagePixelSigma()/focus_length;
    const double epl_threshold = options_.epl_dist2_threshold*pixel_usigma*pixel_usigma;
    const double px_threshold = options_.pixel_error_threshold*pixel_usigma;

    KeyFrame::Ptr reference_keyframe = keyframe->getRefKeyFrame();
    std::set<KeyFrame::Ptr> connect_keyframes = reference_keyframe->getConnectedKeyFrames(num);
    connect_keyframes.insert(reference_keyframe);

    if(connect_keyframes.empty())
        return 0;

    int matched_count = 0;
    for(const KeyFrame::Ptr &kf : connect_keyframes)
    {
        int matched_count_cur = reprojectSeeds(keyframe, kf, epl_threshold, options_.align_epslion*px_threshold, false);

        matched_count += matched_count_cur;
        if(matched_count_cur == 0)
            break;
    }

    return matched_count;
}

int DepthFilter::reprojectSeeds(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame, double epl_threshold, double pixel_error, bool created)
{
    std::vector<size_t> indices = keyframe->getSeedCreateIndices();
    if(indices.empty()) return 0;
    std::vector<Seed::Ptr> seeds = keyframe->getSeeds();

    SE3d T_cur_from_ref = frame->Tcw() * keyframe->pose();

    int project_count = 0;
    int matched_count = 0;
    Vector2d px_matched;
    int level_matched;
    for(const size_t &idx: indices)
    {
        const Seed::Ptr &seed = seeds[idx];
//        if(frame->hasSeed(seed))
//            continue;

        project_count++;
        bool matched = findEpipolarMatch(seed, keyframe, frame, T_cur_from_ref, px_matched, level_matched);
        if(!matched)
            continue;

        //! check distance to epl, incase of the aligen draft
        Vector2d fn_matched = frame->cam_->lift(px_matched).head<2>();
        double dist2 = utils::Fundamental::computeErrorSquared(keyframe->pose().translation(), seed->fn_ref/seed->getInvDepth(), T_cur_from_ref, fn_matched);
        if(dist2 > epl_threshold)
            continue;

        double pixel_disparity = (seed->px_ref - px_matched).norm() * Frame::inv_scale_factors_.at(level_matched);//seed->level_ref);
        if(pixel_disparity < options_.min_pixel_disparity)
            continue;

        double depth = -1;
        const Vector3d fn_cur = frame->cam_->lift(px_matched);
        bool succeed = utils::triangulate(T_cur_from_ref.rotationMatrix(), T_cur_from_ref.translation(), seed->fn_ref, fn_cur, depth);
        if(!succeed)
            continue;

//            double tau = seed->computeTau(T_ref_from_cur, seed->fn_ref, depth, px_error_angle);
        double tau = seed->computeVar(T_cur_from_ref, depth, pixel_error);
//            tau = tau + Config::pixelUnSigma();
        seed->update(1.0/depth, tau*tau);

        //! check converge
        if(seed->checkConvergence())
        {
            seed_converged_callback_(seed);
            continue;
        }

        //! update px
        if(created)
        {
            Feature::Ptr new_ft = Feature::create(px_matched, fn_cur, level_matched);
            frame->addSeedFeatureMatch(seed, new_ft);
        }
        matched_count++;
    }

    return matched_count;
}

int DepthFilter::reprojectAllSeeds(const Frame::Ptr &frame)
{
    static double focus_length = MIN(frame->cam_->fx(), frame->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma()/focus_length;
    static double epl_threshold = options_.epl_dist2_threshold*pixel_usigma*pixel_usigma;
    static double px_threshold = options_.pixel_error_threshold*pixel_usigma;
//    //! get new seeds for track
//    std::set<KeyFrame::Ptr> candidate_keyframes = frame->getRefKeyFrame()->getConnectedKeyFrames(options_.max_kfs);
//    candidate_keyframes.insert(frame->getRefKeyFrame());

    int matched_count = 0;
    for(const KeyFrame::Ptr &kf : seeds_buffer_)
    {
        matched_count += reprojectSeeds(kf, frame, epl_threshold, px_threshold);
    }

    return matched_count;
}

bool DepthFilter::findEpipolarMatch(const Seed::Ptr &seed,
                                    const KeyFrame::Ptr &keyframe,
                                    const Frame::Ptr &frame,
                                    const SE3d &T_cur_from_ref,
                                    Vector2d &px_matched,
                                    int &level_matched)
{
    static const int PatchSize = AlignPatch8x8::Size;
    static const int HalfPatchSize = AlignPatch8x8::HalfSize;

    //! check if in the view of current frame
    const double z_ref = 1.0/seed->getInvDepth();
    const Vector3d xyz_ref(seed->fn_ref * z_ref);
    const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    const double z_cur = xyz_cur[2];
    if(z_cur < 0.001f)
        return false;

    //! d - inverse depth, z - depth
    const double sigma = std::sqrt(seed->getVariance());
    const double d_max = z_ref + sigma;
    const double d_min = MAX(z_ref- sigma, 0.00000001f);
    const double z_ref_min = 1.0/d_max;
    const double z_ref_max = 1.0/d_min;

    //! calculate best search level
    const int level_ref = seed->level_ref;
    const int level_cur = MapPoint::predictScale(z_ref, z_cur, level_ref, Frame::nlevels_-1);
    level_matched = level_cur;
    const double inv_scale_cur = Frame::inv_scale_factors_.at(level_cur);

    const Vector2d px_cur(frame->cam_->project(xyz_cur) * inv_scale_cur);
    if(!frame->cam_->isInFrame(px_cur.cast<int>(), HalfPatchSize, inv_scale_cur))
        return false;

    //! px in image plane
    Vector3d xyz_near = T_cur_from_ref * (seed->fn_ref * z_ref_min);
    Vector3d xyz_far  = T_cur_from_ref * (seed->fn_ref * z_ref_max);

    //! Pc = R*Pr + t = z * R*Pn + t
    if(xyz_near[2] < 0.001f)
    {
        const Vector3d t = T_cur_from_ref.translation();
        const Vector3d R_fn = T_cur_from_ref.rotationMatrix() * seed->fn_ref;
        double z_ref_min_adjust = (0.001f - t[2]) / R_fn[2];
        xyz_near = z_ref_min_adjust * R_fn + t;
    }

    Vector2d px_near = frame->cam_->project(xyz_near) * inv_scale_cur;
    Vector2d px_far  = frame->cam_->project(xyz_far) * inv_scale_cur;
    Vector2d epl_px_dir(px_near - px_far);
    epl_px_dir.normalize();

    //! make search pixel all within image
    const int min_sample_width = HalfPatchSize;
    const int min_sample_height = HalfPatchSize;
    const int max_sample_width = frame->cam_->width() * inv_scale_cur;
    const int max_sample_height = frame->cam_->height() * inv_scale_cur;
    if(px_near[0] <= min_sample_width)
    {
        double adjust = ceil(min_sample_width - px_near[0])/epl_px_dir[0];
        px_near.noalias() += adjust * epl_px_dir;
    }
    else if(px_near[0] >= max_sample_width)
    {
        double adjust = floor(max_sample_width - px_near[0])/epl_px_dir[0];
        px_near.noalias() += adjust * epl_px_dir;
    }

    if(px_near[1] <= min_sample_height)
    {
        double adjust = ceil(min_sample_height - px_near[1])/epl_px_dir[1];
        px_near.noalias() += adjust * epl_px_dir;
    }
    else if(px_near[1] >= max_sample_height)
    {
        double adjust = floor(max_sample_height - px_near[1])/epl_px_dir[1];
        px_near.noalias() += adjust * epl_px_dir;
    }

    if(px_near[0] <= min_sample_width || px_near[0] >= max_sample_width ||
        px_near[1] <= min_sample_height || px_near[1] >= max_sample_height)
        return false;

    if(px_far[0] <= min_sample_width)
    {
        double adjust = ceil(min_sample_width - px_far[0])/epl_px_dir[0];
        px_far.noalias() += adjust * epl_px_dir;
    }
    else if(px_far[0] >= max_sample_width)
    {
        double adjust = floor(max_sample_width - px_far[0])/epl_px_dir[0];
        px_far.noalias() += adjust * epl_px_dir;
    }

    if(px_far[1] <= min_sample_height)
    {
        double adjust = ceil(min_sample_height - px_far[1])/epl_px_dir[1];
        px_far.noalias() += adjust * epl_px_dir;
    }
    else if(px_far[1] >= max_sample_height)
    {
        double adjust = floor(max_sample_height - px_far[1])/epl_px_dir[1];
        px_far.noalias() += adjust * epl_px_dir;
    }

    if(px_far[0] <= min_sample_width || px_far[0] >= max_sample_width ||
        px_far[1] <= min_sample_height || px_far[1] >= max_sample_height)
        return false;

    //! reject the seed whose epl is too long
    //! it maybe not nessary
    double epl_length = (px_near-px_far).norm();
    if(epl_length > options_.max_epl_length)
        return false;

    //! get px in normilzed plane
    xyz_near = frame->cam_->lift(px_near / inv_scale_cur);
    xyz_far = frame->cam_->lift(px_far / inv_scale_cur);
    Vector2d epl_dir = (xyz_near - xyz_far).head<2>();

    //! get warp patch
    Matrix2d A_cur_from_ref;

    utils::getWarpMatrixAffine(keyframe->cam_, frame->cam_, seed->px_ref, seed->fn_ref, level_ref,
                               z_ref, T_cur_from_ref, PatchSize, A_cur_from_ref);

//    double det = A_cur_from_ref.determinant() / factor;
//    std::cout << "***** det: " <<  det << std::endl;

    static const int patch_border_size = PatchSize+2;
    cv::Mat image_ref = keyframe->getImage(level_ref);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                seed->px_ref, level_ref, level_cur);

    Matrix<float, PatchSize, PatchSize, RowMajor> patch;
    patch = patch_with_border.block(1, 1, PatchSize, PatchSize);

    const cv::Mat image_cur = frame->getImage(level_cur);

    Vector2d px_best(-1,-1);
    Vector3d estimate(0,0,0);
    if(epl_length > 2.0)
    {
        int n_steps = epl_length / 0.707;
        Vector2d step = epl_dir / n_steps;

        // TODO 使用模板来加速！！！
        //! SSD
        double t0 = (double)cv::getTickCount();
        ZSSD<float, PatchSize> zssd(patch);
        double score_best = std::numeric_limits<double>::max();
        double score_second = score_best;
        int index_best = -1;
        int index_second = -1;

        Vector2d fn_start = xyz_far.head<2>() - step * 2;
        n_steps += 2;
        Vector2d fn(fn_start);
        for(int i = 0; i < n_steps; ++i, fn += step)
        {
            Vector2f px = (frame->cam_->project(fn[0], fn[1]) * inv_scale_cur).cast<float>();

            //! always in frame's view
            if(!frame->cam_->isInFrame(px.cast<int>(), HalfPatchSize, inv_scale_cur))
                continue;

            Matrix<float, PatchSize, PatchSize, RowMajor> patch_cur;
            utils::interpolateMat<uchar, float, PatchSize>(image_cur, patch_cur, px[0], px[1]);

            float score = zssd.compute_score(patch_cur);

            if(score < score_best)
            {
                score_second = score_best;
                index_second = index_best;
                score_best = score;
                index_best = i;
            }
            else if(score < score_second)
            {
                score_second = score;
                index_second = i;
            }
        }

        if(score_best > 0.8 * score_second && std::abs(index_best - index_second) > 3)
            return false;

        if(score_best > zssd.threshold())
            return false;

        Vector2d pn_best = fn_start + index_best * step;
        px_best = frame->cam_->project(pn_best[0], pn_best[1]) * inv_scale_cur;

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

    estimate.head<2>() = px_best;
    if(!AlignPatch8x8::align2DI(image_cur, patch_with_border, estimate, 30, options_.align_epslion, verbose_))
    {
//        std::cout << "dx:\n " << dx << std::endl;
//        std::cout << "dy:\n " << dy << std::endl;

        static bool show = false;
        if(show)
        {
            //        showMatch(keyframe->getImage(level_ref), current_frame_->getImage(level_cur), px_near, px_far, ft->px/factor, px_best);
//            DISPLAY:
            showEplMatch(keyframe, frame, T_cur_from_ref, level_ref, level_cur, xyz_near, xyz_far, xyz_ref, px_best);
            FeatureTracker::showAffine(keyframe->getImage(level_ref), seed->px_ref * inv_scale_cur, A_cur_from_ref.inverse(), 8, level_ref);
        }

        return false;
    }

    //! transform to level-0
    px_matched = estimate.head<2>() / inv_scale_cur;

    LOG_IF(INFO, verbose_) << "Found! [" << seed->px_ref.transpose() << "] "
                           << "dst: [" << px_matched.transpose() << "] "
                           << "epl: [" << px_near.transpose() << "]--[" << px_far.transpose() << "]" << std::endl;

//    showMatch(image_ref, image_cur, px_near, px_far, seed.ft->px/factor, px_matched/factor);

    return true;
}

}
