#include <future>
#include "config.hpp"
#include "utils.hpp"
#include "depth_filter.hpp"
#include "feature_alignment.hpp"
#include "image_alignment.hpp"

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
//! Seed
Seed::Seed(const KeyFrame::Ptr &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min) :
    kf(kf), fn_ref(fn), px_ref(px), level_ref(level), px_cur(px), level_cur(level),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range)
{
    assert(fn_ref[2] == 1);
}

double Seed::computeTau(
    const SE3d& T_ref_cur,
    const Vector3d& f,
    const double z,
    const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double f_norm = f.norm();
    double alpha = acos(f.dot(t)/(t_norm*f_norm)); // dot product
    double beta = acos(-a.dot(t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    z_plus /= f_norm;

    double tau = z_plus - z;
    return 0.5 * (1.0/MAX(0.0000001, z-tau) - 1.0/(z+tau));
}

double Seed::computeVar(const SE3d &T_cur_ref, const double z, const double delta)
{
    const Vector3d &t(T_cur_ref.translation()); // from cur->ref in cur's frame
    Vector3d xyz_r(fn_ref*z);
    Vector3d f_c(T_cur_ref * xyz_r);
    Vector3d f_r(f_c-t);

    double t_norm = t.norm();
    double f_c_norm = f_c.norm();
    double f_r_norm = f_r.norm();

//    double alpha = acos(f_r.dot(-t)/f_r_norm/t_norm);

    double epslion = atan(0.5*delta/f_c_norm/f_c[2])*2.0;
//    epslion  = 0.0021867665614925609;
    double beta = acos(f_c.dot(t)/(f_c_norm*t_norm));
    double gamma = acos(f_c.dot(f_r)/(f_c_norm*f_r_norm));

    double z1 = t_norm * sin(beta+epslion) / sin(gamma-epslion);
    z1 /= f_r_norm;

    return 0.5 * (1.0/MAX(0.0000001, 2*z-z1) - 1.0/(z1));
}

void Seed::update(const double x, const double tau2)
{
    double norm_scale = sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;

    double s2 = 1./(1./sigma2 + 1./tau2);
    double m = s2*(mu/sigma2 + x/tau2);
    double C1 = a/(a+b) * utils::normal_distribution<double>(x, mu, norm_scale);
    double C2 = b/(a+b) * 1./z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    double e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));

    // update parameters
    double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e-f)/(f-e/f);
    b = a*(1.0f-f)/f;

    history.emplace_back(x, 1.0/mu);
}

//! =================================================================================================
//! DepthFilter
DepthFilter::DepthFilter(const FastDetector::Ptr &fast_detector, const Callback &callback, bool report, bool verbose) :
    seed_coverged_callback_(callback), fast_detector_(fast_detector),
    report_(report), verbose_(report&&verbose), track_thread_enabled_(false)

{
    options_.max_kfs = 3;
    options_.max_epl_length = 1000;
    options_.epl_dist2_threshold = 16 * Config::pixelUnSigma2();
    options_.seed_converge_threshold = 1.0/200.0;
    options_.klt_epslion = 0.0001;
    options_.align_epslion = 0.0001;
    options_.min_disparity = 100;
    options_.min_track_features = 50;
}

void DepthFilter::setMap(const Map::Ptr &map)
{
    map_ = map;
}

void DepthFilter::drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst)
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
        Vector2d ft_px = ft->px_;
        cv::Point2f px(ft_px[0], ft_px[1]);
        cv::Scalar color(0, 255, 0);
        cv::circle(dst, px, 2, color, -1);

        string id_str = std::to_string((frame->Tcw()*ft->mpt_->pose()).norm());//ft->mpt_->getFoundRatio());//
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

void DepthFilter::enableTrackThread()
{
    track_thread_enabled_ = true;
}

void DepthFilter::disableTrackThread()
{
    track_thread_enabled_ = false;
}

void DepthFilter::startMainThread()
{
    if(filter_thread_ == nullptr)
        filter_thread_ = std::make_shared<std::thread>(std::bind(&DepthFilter::run, this));
}

void DepthFilter::stopMainThread()
{
    setStop();
    filter_thread_->join();
    filter_thread_.reset();
}

void DepthFilter::insertFrame(const Frame::Ptr &frame)
{
    LOG_ASSERT(frame != nullptr) << "[Mapping] Error input! Frame should not be null!";
    last_frame_ = current_frame_;
    current_frame_ = frame;
    if(track_thread_enabled_)
    {
        seeds_track_future_ = std::async(std::launch::async, &DepthFilter::trackSeeds, this);
    }
    else
    {
        trackSeeds();
    }
}

void DepthFilter::finishFrame()
{
    if(filter_thread_ == nullptr)
    {
        //! wait if trackSeeds() did not finish
        if(track_thread_enabled_)
        {
            seeds_track_future_.wait();
            uint64_t id = seeds_track_future_.get();
            LOG_ASSERT( id == current_frame_->id_) << "[Mapping] Wrong frame input! " << id << " != " << current_frame_->id_;
        }

        double t0 = (double)cv::getTickCount();
        updateSeeds();
        double t1 = (double)cv::getTickCount();
        reprojectSeeds();
        double t2 = (double)cv::getTickCount();
        LOG(ERROR) << " ReprojectSeeds Time: " << (t1-t0)/cv::getTickFrequency() << " " << (t2-t1)/cv::getTickFrequency();
    }
    else
    {
        cond_process_.notify_one();
//        frames_buffer_.emplace_back()
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

bool DepthFilter::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_frame_);
        cond_process_.wait_for(lock, std::chrono::milliseconds(5));
        if(frames_buffer_.empty())
            return false;

//        current_frame_ = frames_buffer_.front();
//        current_depth_ = depth_buffer_.front();
//        frames_buffer_.pop_front();
//        depth_buffer_.pop_front();
    }

    return true;
}

void DepthFilter::run()
{
    while(!isRequiredStop())
    {
        if(!checkNewFrame())
            continue;

    }
}

int DepthFilter::createSeeds(const KeyFrame::Ptr &kf)
{
    if(kf == nullptr)
        return 0;

    std::vector<Feature::Ptr> fts;
    kf->getFeatures(fts);

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px_[0], ft->px_[1], 0, ft->level_));
    }

    // TODO 如果对应的seed收敛，在跟踪过的关键帧增加观测？
    if(current_frame_ != nullptr && kf->frame_id_ == current_frame_->id_)
    {
        for(const Seed::Ptr &seed : tracked_seeds_)
        {
            const Vector2d &px = seed->px_cur;
            old_corners.emplace_back(Corner(px[0], px[1], seed->level_cur, seed->level_cur));
        }
    }

    Corners new_corners;
    fast_detector_->detect(kf->images(), new_corners, old_corners, 150);

    if(new_corners.empty())
        return 0;

    double depth_mean;
    double depth_min;
    kf->getSceneDepth(depth_mean, depth_min);
    Seeds new_seeds;
    for(const Corner &corner : new_corners)
    {
        const Vector2d px(corner.x, corner.y);
        const Vector3d fn(kf->cam_->lift(px));
        new_seeds.emplace_back(Seed::create(kf, px, fn, corner.level, depth_mean, depth_min));
    }
    seeds_buffer_.emplace_back(kf, std::make_shared<Seeds>(new_seeds));

    for(const Seed::Ptr &seed : new_seeds)
        tracked_seeds_.emplace_back(seed);

    return (int)new_seeds.size();
}

uint64_t DepthFilter::trackSeeds()
{
    if(current_frame_ == nullptr || last_frame_ == nullptr)
        return 0;

    //! track seeds by klt
    std::vector<cv::Point2f> pts_to_track;
    pts_to_track.reserve(tracked_seeds_.size());
    for(const Seed::Ptr &seed : tracked_seeds_)
    {
        pts_to_track.emplace_back(cv::Point2f((float)seed->px_cur[0], (float)seed->px_cur[1]));
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
            disparity.push_back(dist);
            track_seeds_itr++;
            count++;
        }
        else
        {
            track_seeds_itr = tracked_seeds_.erase(track_seeds_itr);
        }
    }

    LOG_IF(INFO, report_) << "[DepthFilter][1] Tracking seeds: " << count;

    return current_frame_->id_;
}

int DepthFilter::updateSeeds()
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
                    seed_coverged_callback_(seed);
                    earseSeed(seed->kf, seed);
                    continue;
                }
            }

            tracked_seeds_.emplace_back(seed);
        }
    }

    LOG_IF(INFO, report_) << "[DepthFilter][2] Seeds updated: " << tracked_seeds_.size();
    return (int)tracked_seeds_.size();
}

int DepthFilter::reprojectSeeds()
{
    //! get new seeds for track
    std::set<KeyFrame::Ptr> candidate_keyframes = current_frame_->getRefKeyFrame()->getConnectedKeyFrames(options_.max_kfs);
    candidate_keyframes.insert(current_frame_->getRefKeyFrame());

    std::set<Seed::Ptr> seed_tracking;
    for(const Seed::Ptr &seed : tracked_seeds_)
        seed_tracking.insert(seed);

    int project_count = 0;
    int matched_count = 0;
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

            project_count++;
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
                seed_coverged_callback_(seed);
                seed_itr = seeds.erase(seed_itr);
                continue;
            }

            //! update px
            seed->px_cur = px_matched;
            tracked_seeds_.emplace_back(seed);
            matched_count++;
            seed_itr++;
        }

        if(seeds.empty())
            buffer_itr = seeds_buffer_.erase(buffer_itr);
        else
            buffer_itr++;
    }

    LOG_IF(ERROR, report_) << "[DepthFilter][3] Seeds reprojected: " << project_count << ", matched: " << matched_count << ", tracked seeds: "<< tracked_seeds_.size();

    return (int) tracked_seeds_.size();
}

bool DepthFilter::earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed)
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

bool DepthFilter::findEpipolarMatch(const Seed::Ptr &seed,
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
    if(z_cur < 0.001f)
        return false;

    //! d - inverse depth, z - depth
    const double sigma = std::sqrt(seed->sigma2);
    const double d_max = z_ref + sigma;
    const double d_min = MAX(z_ref- sigma, 0.00000001f);
    const double z_ref_min = 1.0/d_max;
    const double z_ref_max = 1.0/d_min;

    //! calculate best search level
    const int level_ref = seed->level_ref;
    const int level_cur = MapPoint::predictScale(z_ref, z_cur, level_ref, frame->max_level_);
    seed->level_cur = level_cur;
    const double scale_cur = 1.0 / (1 << level_cur);

    const Vector2d px_cur(frame->cam_->project(xyz_cur) * scale_cur);
    if(!frame->cam_->isInFrame(px_cur.cast<int>(), half_patch_size, level_cur))
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

    Vector2d px_near = frame->cam_->project(xyz_near) * scale_cur;
    Vector2d px_far  = frame->cam_->project(xyz_far) * scale_cur;
    Vector2d epl_px_dir(px_near - px_far);
    epl_px_dir.normalize();

    //! make search pixel all within image
    const int min_sample_width = half_patch_size;
    const int min_sample_height = half_patch_size;
    const int max_sample_width = frame->cam_->width() * scale_cur;
    const int max_sample_height = frame->cam_->height() * scale_cur;
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
    xyz_near = frame->cam_->lift(px_near / scale_cur);
    xyz_far = frame->cam_->lift(px_far / scale_cur);
    Vector2d epl_dir = (xyz_near - xyz_far).head<2>();

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
    Vector3d estimate(0,0,0);
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
            Vector2f px = (frame->cam_->project(fn[0], fn[1]) * scale_cur).cast<float>();

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
        px_best = frame->cam_->project(pn_best[0], pn_best[1]) * scale_cur;

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
    if(!AlignPatch::align2DI(image_cur, patch_with_border, estimate, 30, options_.align_epslion, verbose_))
    {
//        std::cout << "dx:\n " << dx << std::endl;
//        std::cout << "dy:\n " << dy << std::endl;

        static bool show = false;
        if(show)
        {
            //        showMatch(keyframe->getImage(level_ref), current_frame_->getImage(level_cur), px_near, px_far, ft->px/factor, px_best);
//            DISPLAY:
            showEplMatch(keyframe, frame, T_cur_from_ref, level_ref, level_cur, xyz_near, xyz_far, xyz_ref, px_best);
            showAffine(keyframe->getImage(level_ref), seed->px_ref * scale_cur, A_cur_from_ref.inverse(), 8, level_ref);
        }

        return false;
    }

    //! transform to level-0
    px_matched = estimate.head<2>() / scale_cur;

    LOG_IF(INFO, verbose_) << "Found! [" << seed->px_ref.transpose() << "] "
                           << "dst: [" << px_matched.transpose() << "] "
                           << "epl: [" << px_near.transpose() << "]--[" << px_far.transpose() << "]" << std::endl;

//    showMatch(image_ref, image_cur, px_near, px_far, seed.ft->px/factor, px_matched/factor);

    return true;
}

}
