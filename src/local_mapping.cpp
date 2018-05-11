#include "config.hpp"
#include "local_mapping.hpp"
#include "feature_alignment.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "image_alignment.hpp"
#include "optimizer.hpp"
#include "time_tracing.hpp"
#include "brief.hpp"

#ifdef SSVO_DBOW_ENABLE
#include <DBoW3/DescManip.h>
#endif

namespace ssvo{

std::ostream& operator<<(std::ostream& out, const Feature& ft)
{
    Vector3d xyz = ft.mpt_->pose();
    out << "{ px: [" << ft.px_[0] << ", " << ft.px_[1] << "],"
        << " fn: [" << ft.fn_[0] << ", " << ft.fn_[1] << ", " << ft.fn_[2] << "],"
        << " level: " << ft.level_
        << " mpt: " << ft.mpt_->id_ << ", [" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << "] "
        << " }";

    return out;
}

TimeTracing::Ptr mapTrace = nullptr;

//! LocalMapper
LocalMapper::LocalMapper(bool report, bool verbose) :
    report_(report), verbose_(report&&verbose),
    mapping_thread_(nullptr), stop_require_(false)
{
    map_ = Map::create();

    options_.min_disparity = 100;
    options_.min_redundant_observations = 3;
    options_.max_features = Config::minCornersPerKeyFrame();
    options_.num_reproject_kfs = MAX(Config::maxReprojectKeyFrames(), 2);
    options_.num_local_ba_kfs = MAX(Config::maxLocalBAKeyFrames(), 1);
    options_.min_local_ba_connected_fts = Config::minLocalBAConnectedFts();
    options_.num_align_iter = 15;
    options_.max_align_epsilon = 0.01;
    options_.max_align_error2 = 3.0;
    options_.min_found_ratio_ = 0.15;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("local_ba");
    time_names.push_back("reproj");
    time_names.push_back("dbow");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("keyframe_id");
    log_names.push_back("num_reproj_kfs");
    log_names.push_back("num_reproj_mpts");
    log_names.push_back("num_matched");
    log_names.push_back("num_fusion");


    string trace_dir = Config::timeTracingDirectory();
    mapTrace.reset(new TimeTracing("ssvo_trace_map", trace_dir, time_names, log_names));


#ifdef SSVO_DBOW_ENABLE
    std::string voc_dir = Config::DBoWDirectory();
    LOG_ASSERT(!voc_dir.empty()) << "Please check the config file! The DBoW directory is not set!";
    vocabulary_ = DBoW3::Vocabulary(voc_dir);
    LOG_ASSERT(!vocabulary_.empty()) << "Please check the config file! The Voc is empty!";
    database_ = DBoW3::Database(vocabulary_, true, 4);

    const int nlevel = Config::imageTopLevel() + 1;
    const int cols = Config::imageWidth();
    const int rows = Config::imageHeight();
    border_tl_.resize(nlevel);
    border_br_.resize(nlevel);

    for(int i = 0; i < nlevel; i++)
    {
        border_tl_[i].x = BRIEF::EDGE_THRESHOLD;
        border_tl_[i].y = BRIEF::EDGE_THRESHOLD;
        border_br_[i].x = cols/(1<<i) - BRIEF::EDGE_THRESHOLD;
        border_br_[i].y = rows/(1<<i) - BRIEF::EDGE_THRESHOLD;
    }

#endif

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
        fts_ref[i]->mpt_->addObservation(keyframe_ref, fts_ref[i]);
        fts_cur[i]->mpt_->addObservation(keyframe_cur, fts_cur[i]);
    }

    for(const Feature::Ptr &ft : fts_ref)
    {
        map_->insertMapPoint(ft->mpt_);
        ft->mpt_->resetType(MapPoint::STABLE);
        ft->mpt_->updateViewAndDepth();
//        addOptimalizeMapPoint(ft->mpt_);
    }

    keyframe_ref->setRefKeyFrame(keyframe_cur);
    keyframe_cur->setRefKeyFrame(keyframe_ref);
    keyframe_ref->updateConnections();
    keyframe_cur->updateConnections();
    insertKeyFrame(keyframe_ref);
    insertKeyFrame(keyframe_cur);

    LOG_IF(INFO, report_) << "[Mapper] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

void LocalMapper::startMainThread()
{
    if(mapping_thread_ == nullptr)
        mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::stopMainThread()
{
    setStop();
    if(mapping_thread_)
    {
        if(mapping_thread_->joinable())
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
        KeyFrame::Ptr keyframe_cur = checkNewKeyFrame();
        if(keyframe_cur)
        {
            mapTrace->startTimer("total");
            std::list<MapPoint::Ptr> bad_mpts;
            int new_seed_features = 0;
            int new_local_features = 0;
            if(map_->kfs_.size() > 2)
            {
//                new_seed_features = createFeatureFromSeedFeature(keyframe_cur);
                mapTrace->startTimer("reproj");
                new_local_features = createFeatureFromLocalMap(keyframe_cur, options_.num_reproject_kfs);
                mapTrace->stopTimer("reproj");
                LOG_IF(INFO, report_) << "[Mapper] create " << new_seed_features << " features from seeds and " << new_local_features << " from local map.";

                mapTrace->startTimer("local_ba");
                Optimizer::localBundleAdjustment(keyframe_cur, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
                mapTrace->stopTimer("local_ba");
            }

            for(const MapPoint::Ptr &mpt : bad_mpts)
            {
                map_->removeMapPoint(mpt);
            }

            checkCulling(keyframe_cur);

            mapTrace->startTimer("dbow");
            addToDatabase(keyframe_cur);
            mapTrace->stopTimer("dbow");

            mapTrace->stopTimer("total");
            mapTrace->writeToFile();

            keyframe_last_ = keyframe_cur;
        }
    }
}

KeyFrame::Ptr LocalMapper::checkNewKeyFrame()
{
    std::unique_lock<std::mutex> lock(mutex_keyframe_);
    cond_process_.wait_for(lock, std::chrono::milliseconds(5));
    if(keyframes_buffer_.empty())
        return nullptr;

    KeyFrame::Ptr keyframe = keyframes_buffer_.front();
    keyframes_buffer_.pop_front();

    return keyframe;
}

void LocalMapper::insertKeyFrame(const KeyFrame::Ptr &keyframe)
{
    //! incase add the same keyframe twice
    if(!map_->insertKeyFrame(keyframe))
        return;;

    mapTrace->log("frame_id", keyframe->frame_id_);
    mapTrace->log("keyframe_id", keyframe->id_);
    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_keyframe_);
        keyframes_buffer_.push_back(keyframe);
        cond_process_.notify_one();
    }
    else
    {
        mapTrace->startTimer("total");
        std::list<MapPoint::Ptr> bad_mpts;
        int new_seed_features = 0;
        int new_local_features = 0;
        if(map_->kfs_.size() > 2)
        {
//            new_seed_features = createFeatureFromSeedFeature(keyframe);
            mapTrace->startTimer("reproj");
            new_local_features = createFeatureFromLocalMap(keyframe, options_.num_reproject_kfs);
            mapTrace->stopTimer("reproj");
            LOG_IF(INFO, report_) << "[Mapper] create " << new_seed_features << " features from seeds and " << new_local_features << " from local map.";

            mapTrace->startTimer("local_ba");
            Optimizer::localBundleAdjustment(keyframe, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
            mapTrace->stopTimer("local_ba");
        }

        for(const MapPoint::Ptr &mpt : bad_mpts)
        {
            map_->removeMapPoint(mpt);
        }

        checkCulling(keyframe);

        addToDatabase(keyframe);

        mapTrace->stopTimer("total");
        mapTrace->writeToFile();

        keyframe_last_ = keyframe;
    }
}

void LocalMapper::finishLastKeyFrame()
{
//    DepthFilter::updateByConnectedKeyFrames(keyframe_last_, 3);
}

void LocalMapper::createFeatureFromSeed(const Seed::Ptr &seed)
{
    //! create new feature
    MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->getInvDepth()));
    Feature::Ptr ft = Feature::create(seed->px_ref, seed->fn_ref, seed->level_ref, mpt);
    seed->kf->addFeature(ft);
    map_->insertMapPoint(mpt);
    mpt->addObservation(seed->kf, ft);
    mpt->updateViewAndDepth();

    std::set<KeyFrame::Ptr> local_keyframes = seed->kf->getConnectedKeyFrames(10);

    for(const KeyFrame::Ptr &kf : local_keyframes)
    {
        Vector3d xyz_cur(kf->Tcw() * mpt->pose());
        if(xyz_cur[2] < 0.0f)
            continue;

        Vector2d px_cur(kf->cam_->project(xyz_cur));
        if(!kf->cam_->isInFrame(px_cur.cast<int>(), 8))
            continue;

        int level_cur = 0;
        const Vector2d px_cur_last = px_cur;
        int result = FeatureTracker::reprojectMapPoint(kf, mpt, px_cur, level_cur, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2);
        if(result != 1)
            continue;

        double error = (px_cur_last-px_cur).norm();
        if(error > 2.0)
            continue;

        Vector3d ft_cur = kf->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        kf->addFeature(new_feature);

        mpt->addObservation(kf, new_feature);
    }

    mpt->updateViewAndDepth();

    if(mpt->observations() > 1)
        Optimizer::refineMapPoint(mpt, 10, true);
}

int LocalMapper::createFeatureFromSeedFeature(const KeyFrame::Ptr &keyframe)
{
    std::vector<Feature::Ptr> seeds;
    keyframe->getSeeds(seeds);

    for(const Feature::Ptr & ft_seed : seeds)
    {
        const Seed::Ptr &seed = ft_seed->seed_;
        MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->getInvDepth()));

        Feature::Ptr ft_ref = Feature::create(seed->px_ref, seed->fn_ref, seed->level_ref, mpt);
        Feature::Ptr ft_cur = Feature::create(ft_seed->px_, keyframe->cam_->lift(ft_seed->px_), ft_seed->level_, mpt);
        seed->kf->addFeature(ft_ref);
        keyframe->addFeature(ft_cur);
        keyframe->removeSeed(seed);

        map_->insertMapPoint(mpt);
        mpt->addObservation(seed->kf, ft_ref);
        mpt->addObservation(keyframe, ft_cur);

        mpt->updateViewAndDepth();
//        addOptimalizeMapPoint(mpt);
    }

    return (int) seeds.size();
}

int LocalMapper::createFeatureFromLocalMap(const KeyFrame::Ptr &keyframe, const int num)
{
    std::set<KeyFrame::Ptr> local_keyframes = keyframe->getConnectedKeyFrames(num);

    std::unordered_set<MapPoint::Ptr> local_mpts;
    MapPoints mpts_cur;
    keyframe->getMapPoints(mpts_cur);
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

            if(mpt->observations() == 1 && mpt->getFoundRatio() < options_.min_found_ratio_)
            {
                mpt->setBad();
                map_->removeMapPoint(mpt);
                continue;
            }

            candidate_mpts.insert(mpt);
        }
    }

    const int max_new_count = options_.max_features * 1.5 - mpts_cur.size();
    //! match the mappoints from nearby keyframes
    int project_count = 0;
    std::list<Feature::Ptr> new_fts;
    for(const MapPoint::Ptr &mpt : candidate_mpts)
    {
        Vector3d xyz_cur(keyframe->Tcw() * mpt->pose());
        if(xyz_cur[2] < 0.0f)
            continue;

        Vector2d px_cur(keyframe->cam_->project(xyz_cur));
        if(!keyframe->cam_->isInFrame(px_cur.cast<int>(), 8))
            continue;

        project_count++;

        int level_cur = 0;
        int result = FeatureTracker::reprojectMapPoint(keyframe, mpt, px_cur, level_cur, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2);
        if(result != 1)
            continue;

        Vector3d ft_cur = keyframe->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        new_fts.push_back(new_feature);

        if(new_fts.size() > max_new_count)
            break;
    }

    //! check whether the matched corner is near a exsit corner.
    //! firstly, create a mask for exsit corners
    const int cols = keyframe->cam_->width();
    const int rows = keyframe->cam_->height();
    cv::Mat mask(rows, cols, CV_16SC1, -1);
    std::vector<Feature::Ptr> old_fts;
    keyframe->getFeatures(old_fts);
    const int old_fts_size = (int) old_fts.size();
    for(int i = 0; i < old_fts_size; ++i)
    {
        const Vector2i px = old_fts[i]->px_.cast<int>();
        for(int c = -2; c <= 2; ++c)
        {
            int16_t* ptr = mask.ptr<int16_t>(px[1]+c) + px[0];
            ptr[-2] = (int16_t)i;
            ptr[-1] = (int16_t)i;
            ptr[0] = (int16_t)i;
            ptr[1] = (int16_t)i;
            ptr[2] = (int16_t)i;
        }
    }

    //! check whether the mappoint is already exist
    int created_count = 0;
    int fusion_count = 0;
    for(const Feature::Ptr &ft : new_fts)
    {
        const Vector2i px = ft->px_.cast<int>();
        int64_t id = mask.ptr<int16_t>(px[1])[px[0]];
        //! if not occupied, create new feature
        if(id == -1)
        {
            //! create new features
            keyframe->addFeature(ft);
            ft->mpt_->addObservation(keyframe, ft);
            ft->mpt_->increaseVisible(2);
            ft->mpt_->increaseFound(2);
//            addOptimalizeMapPoint(ft->mpt_);
            created_count++;
            LOG_IF(INFO, verbose_) << " create new feature from mpt " << ft->mpt_->id_;
        }
        //! if already occupied, check whether the mappoint is the same
        else
        {
            MapPoint::Ptr mpt_new = ft->mpt_;
            MapPoint::Ptr mpt_old = old_fts[id]->mpt_;

            if(mpt_new == mpt_old) //! rarely happen
                continue;

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
                Vector2d px_delta(ft_new->px_ - ft_old->px_);
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
                    Vector3d xyz_new(kf_new->Tcw() * ft_old->mpt_->pose());
                    if(xyz_new[2] < 0.0f)
                        continue;

                    Vector2d px_new(kf_new->cam_->project(xyz_new));
                    if(!kf_new->cam_->isInFrame(px_new.cast<int>(), 8))
                        continue;

                    int level_new = 0;
                    bool matched = FeatureTracker::trackFeature(kf_old_ref, kf_new, ft_old, px_new, level_new, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_new.find(kf_new)->second, px_new[0], px_new[1], level_new);
                }

                //! update ft if succeed
                const AbstractCamera::Ptr &cam = keyframe->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px_[0] = std::get<1>(it);
                    ft_update->px_[1] = std::get<2>(it);
                    ft_update->level_ = std::get<3>(it);
                    ft_update->fn_ = cam->lift(ft_update->px_);
                }

                //! fusion the mappoint
                //! just reject the new one
                mpt_old->fusion(mpt_new);
                map_->removeMapPoint(mpt_new);

//                addOptimalizeMapPoint(mpt_old);

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

                    Vector3d xyz_old(kf_old->Tcw() * ft_new->mpt_->pose());
                    if(xyz_old[2] < 0.0f)
                        continue;

                    Vector2d px_old(kf_old->cam_->project(xyz_old));
                    if(!kf_old->cam_->isInFrame(px_old.cast<int>(), 8))
                        continue;

                    int level_old = 0;
                    bool matched = FeatureTracker::trackFeature(kf_new_ref, kf_old, ft_new, px_old, level_old, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_old.find(kf_old)->second, px_old[0], px_old[1], level_old);
                }

                //! update ft if succeed
                const AbstractCamera::Ptr &cam = keyframe->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px_[0] = std::get<1>(it);
                    ft_update->px_[1] = std::get<2>(it);
                    ft_update->level_ = std::get<3>(it);
                    ft_update->fn_ = cam->lift(ft_update->px_);
                }

                //! add new feature for keyframe, then fusion the mappoint
                ft->mpt_ = mpt_new;
                keyframe->addFeature(ft);
                mpt_new->addObservation(keyframe, ft);

                mpt_new->fusion(mpt_old);
                map_->removeMapPoint(mpt_old);

//                addOptimalizeMapPoint(mpt_new);

                LOG_IF(INFO, verbose_) << " Fusion mpt " << mpt_new->id_ << " with mpt " << mpt_old->id_;
//                goto SHOW;
            }

            fusion_count++;
            continue;

//            SHOW:
//            std::cout << " mpt_new: " << mpt_new->id_ << ", " << mpt_new->pose().transpose() << std::endl;
//            for(const auto &it : obs_new)
//            {
//                std::cout << "-kf: " << it.first->id_ << " px: [" << it.second->px_[0] << ", " << it.second->px_[1] << "]" << std::endl;
//            }
//
//            std::cout << " mpt_old: " << mpt_old->id_ << ", " << mpt_old->pose().transpose() << std::endl;
//            for(const auto &it : obs_old)
//            {
//                std::cout << "=kf: " << it.first->id_ << " px: [" << it.second->px_[0] << ", " << it.second->px_[1] << "]" << std::endl;
//            }
//
//            for(const auto &it : obs_new)
//            {
//                string name = "new -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level_).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px_[0]/(1<<it.second->level_), it.second->px_[1]/(1<<it.second->level_));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//
//            for(const auto &it : obs_old)
//            {
//                string name = "old -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level_).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px_[0]/(1<<it.second->level_), it.second->px_[1]/(1<<it.second->level_));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//            cv::waitKey(0);
        }

    }

    mapTrace->log("num_reproj_mpts", project_count);
    mapTrace->log("num_reproj_kfs", local_keyframes.size());
    mapTrace->log("num_fusion", fusion_count);
    mapTrace->log("num_matched", created_count);
    LOG_IF(WARNING, report_) << "[Mapper][1] old points: " << mpts_cur.size() << ". All candidate: " << candidate_mpts.size() << ", projected: " << project_count
                             << ", points matched: " << new_fts.size() << " with " << created_count << " created, " << fusion_count << " fusioned. ";

    return created_count;
}

void LocalMapper::addOptimalizeMapPoint(const MapPoint::Ptr &mpt)
{
    std::unique_lock<std::mutex> lock(mutex_optimalize_mpts_);
    optimalize_candidate_mpts_.push_back(mpt);
}

bool mptOptimizeOrder(const MapPoint::Ptr &mpt1, const MapPoint::Ptr &mpt2)
{
    if(mpt1->type() < mpt1->type())
        return true;
    else if(mpt1->type() == mpt1->type())
    {
        if(mpt1->last_structure_optimal_ < mpt1->last_structure_optimal_)
            return true;
    }

    return false;
}

int LocalMapper::refineMapPoints(const int max_optimalize_num)
{
    double t0 = (double)cv::getTickCount();
    static uint64_t optimal_time = 0;
    std::unordered_set<MapPoint::Ptr> mpts_for_optimizing;
    int optilize_num = 0;
    int remain_num = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_optimalize_mpts_);
        optimalize_candidate_mpts_.sort(mptOptimizeOrder);

        optilize_num = max_optimalize_num == -1 ? (int)optimalize_candidate_mpts_.size() : max_optimalize_num;
        for(int i = 0; i < optilize_num && !optimalize_candidate_mpts_.empty(); ++i)
        {
            if(!optimalize_candidate_mpts_.front()->isBad())
                mpts_for_optimizing.insert(optimalize_candidate_mpts_.front());

            optimalize_candidate_mpts_.pop_front();
        }

        std::list<MapPoint::Ptr>::iterator mpt_ptr = optimalize_candidate_mpts_.begin();
        for(; mpt_ptr!=optimalize_candidate_mpts_.end(); mpt_ptr++)
        {
            if(mpts_for_optimizing.count(*mpt_ptr))
            {
                mpt_ptr = optimalize_candidate_mpts_.erase(mpt_ptr);
            }
        }
        remain_num = (int)optimalize_candidate_mpts_.size();
    }

    std::set<KeyFrame::Ptr> changed_keyframes;
    for(const MapPoint::Ptr &mpt:mpts_for_optimizing)
    {
        Optimizer::refineMapPoint(mpt, 10);

        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        double max_residual = Config::imagePixelUnSigma2() * 2;
        for(const auto &item : obs)
        {
            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->pose());
            if(residual < max_residual)
                continue;

            mpt->removeObservation(item.first);
            changed_keyframes.insert(item.first);

            if(mpt->type() == MapPoint::BAD)
                map_->removeMapPoint(mpt);
            else if(mpt->type() == MapPoint::SEED)
                mpt->resetType(MapPoint::STABLE);

            mpt->last_structure_optimal_ = optimal_time;
        }
    }

    optimal_time++;

    for(const KeyFrame::Ptr &kf : changed_keyframes)
    {
        kf->updateConnections();
    }

    double t1 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[Mapper][2] Refine MapPoint Time: " << (t1-t0)*1000/cv::getTickFrequency()
                             << "ms, mpts: " << mpts_for_optimizing.size() << ", remained: " << remain_num;

    return (int)mpts_for_optimizing.size();
}

void LocalMapper::checkCulling(const KeyFrame::Ptr &keyframe)
{

    return;

    const std::set<KeyFrame::Ptr> connected_keyframes = keyframe->getConnectedKeyFrames();

    int count = 0;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        if(kf->id_ == 0 || kf->isBad())
            continue;

        const int observations_threshold = 3;
        int redundant_observations = 0;
        MapPoints mpts;
        kf->getMapPoints(mpts);
        //std::cout << "mpt obs: [";
        for(const MapPoint::Ptr &mpt : mpts)
        {
            std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
            //std::cout << obs.size() << ",";
            if(obs.size() > observations_threshold)
            {
                int observations = 0;
                const Feature::Ptr &ft = obs[kf];
                for(const auto &it : obs)
                {
                    if(it.first == kf)
                        continue;

                    //if(it.second->level_ <= ft->level_+1)
                    {
                        observations++;
                        if(observations >= options_.min_redundant_observations)
                            break;
                    }
                }

                if(observations >= options_.min_redundant_observations)
                    redundant_observations++;
            }

            if(redundant_observations > mpts.size() * 0.8)
            {
                kf->setBad();
                map_->removeKeyFrame(kf);
                count++;
            }
        }
        //std::cout << "]" <<std::endl;

       // std::cout <<"redundant_observations: " << redundant_observations << " mpts: " << mpts.size() << std::endl;

    }
}

template <>
inline size_t Grid<Feature::Ptr>::getIndex(const Feature::Ptr &element)
{
    const Vector2d &px = element->px_;
    return static_cast<size_t>(px[1]/grid_size_)*grid_n_cols_
        + static_cast<size_t>(px[0]/grid_size_);
}

void LocalMapper::addToDatabase(const KeyFrame::Ptr &keyframe)
{
#ifdef SSVO_DBOW_ENABLE
    keyframe->getFeatures(keyframe->dbow_fts_);

    const int cols = keyframe->cam_->width();
    const int rows = keyframe->cam_->height();
    const int N = options_.max_features;
    Grid<Feature::Ptr> grid(cols, rows, 30);

    for(const Feature::Ptr &ft : keyframe->dbow_fts_)
    {
        if(ft->px_[0] <= border_tl_[ft->level_].x ||
            ft->px_[1] <= border_tl_[ft->level_].y ||
            ft->px_[0] >= border_br_[ft->level_].x ||
            ft->px_[1] >= border_br_[ft->level_].y)
            continue;

        grid.insert(ft);
    }

    resetGridAdaptive(grid, N, 20);

    grid.sort();
    grid.getBestElement(keyframe->dbow_fts_);

    std::vector<cv::KeyPoint> kps;
    for(const Feature::Ptr &ft : keyframe->dbow_fts_)
        kps.emplace_back(cv::KeyPoint(ft->px_[0], ft->px_[1], 31, -1, 0, ft->level_));

    BRIEF brief;
    brief.compute(keyframe->images(), kps, keyframe->descriptors_);

    keyframe->dbow_Id_ = database_.add(keyframe->descriptors_, nullptr, nullptr);

    LOG_ASSERT(keyframe->dbow_Id_ == keyframe->id_) << "DBoW Id(" << keyframe->dbow_Id_ << ") is not match the keyframe's Id(" << keyframe->id_ << ")!";

#endif
}

KeyFrame::Ptr LocalMapper::relocalizeByDBoW(const Frame::Ptr &frame, const Corners &corners)
{
    KeyFrame::Ptr reference = nullptr;

#ifdef SSVO_DBOW_ENABLE

    std::vector<cv::KeyPoint> kps;
    for(const Corner & corner : corners)
    {
        if(corner.x <= border_tl_[corner.level].x ||
            corner.y <= border_tl_[corner.level].y ||
            corner.x >= border_br_[corner.level].x ||
            corner.y >= border_br_[corner.level].y)
            continue;

        kps.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));
    }

    BRIEF brief;
    cv::Mat _descriptors;
    brief.compute(frame->images(), kps, _descriptors);
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(_descriptors.rows);
    for(int i = 0; i < _descriptors.rows; i++)
        descriptors.push_back(_descriptors.row(i));

    DBoW3::BowVector bow_vec;
    DBoW3::FeatureVector feat_vec;
    vocabulary_.transform(descriptors, bow_vec, feat_vec, 4);

    DBoW3::QueryResults results;
    database_.query(bow_vec, results, 1);

    if(results.empty())
        return nullptr;

    DBoW3::Result result = results[0];

    reference = map_->getKeyFrame(result.Id);

#endif

    // TODO 如果有关键帧剔除，则数据库索引存在问题。
    if(reference == nullptr)
        return nullptr;

    LOG_ASSERT(reference->dbow_Id_ == reference->id_) << "DBoW Id(" << reference->dbow_Id_ << ") is not match the keyframe's Id(" << reference->id_ << ")!";

    return reference;
}

}