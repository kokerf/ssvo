#include "config.hpp"
#include "local_mapping.hpp"
#include "feature_alignment.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "image_alignment.hpp"
#include "optimizer.hpp"
#include "time_tracing.hpp"

namespace ssvo{

std::ostream& operator<<(std::ostream& out, const Feature& ft)
{
    out << "{ px: [" << ft.px_[0] << ", " << ft.px_[1] << "],"
        << " fn: [" << ft.fn_[0] << ", " << ft.fn_[1] << ", " << ft.fn_[2] << "],"
        << " level: " << ft.corner_.level
        << " }";

    return out;
}

TimeTracing::Ptr mapTrace = nullptr;

//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr fast, bool report, bool verbose) :
    fast_detector_(fast), report_(report), verbose_(report&&verbose),
    mapping_thread_(nullptr), stop_require_(false)
{
    map_ = Map::create();

    brief_ = BRIEF::create(fast->getScaleFactor(), fast->getNLevels());

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
    time_names.push_back("dbow");
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

    std::string voc_dir = Config::DBoWDirectory();
    LOG_ASSERT(!voc_dir.empty()) << "Please check the config file! The DBoW directory is not set!";
    vocabulary_ = DBoW3::Vocabulary(voc_dir);
    LOG_ASSERT(!vocabulary_.empty()) << "Please check the config file! The Voc is empty!";
    database_ = DBoW3::Database(vocabulary_, true, 4);
}

void LocalMapper::createInitalMap(const KeyFrame::Ptr &keyframe_ref, const KeyFrame::Ptr &keyframe_cur)
{
    //! before import, make sure the features are stored in the same order!
    std::vector<MapPoint::Ptr> mpts_ref = keyframe_ref->getMapPoints();
    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<size_t> matches_ref = keyframe_ref->getMapPointMatchIndices();

    std::vector<MapPoint::Ptr> mpts_cur = keyframe_cur->getMapPoints();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();
    std::vector<size_t> matches_cur = keyframe_cur->getMapPointMatchIndices();

    for(const size_t &idx : matches_ref)
    {
        const MapPoint::Ptr &mpt = mpts_ref[idx];
        mpt->addObservation(keyframe_ref, idx);
    }

    for(const size_t &idx : matches_cur)
    {
        const MapPoint::Ptr &mpt = mpts_cur[idx];
        mpt->addObservation(keyframe_cur, idx);

        map_->insertMapPoint(mpt);
        mpt->resetType(MapPoint::STABLE);
        mpt->updateViewAndDepth();
        mpt->computeDistinctiveDescriptor();
    }

    keyframe_ref->setRefKeyFrame(keyframe_cur);
    keyframe_cur->setRefKeyFrame(keyframe_ref);
    keyframe_ref->updateConnections();
    keyframe_cur->updateConnections();

    LOG_IF(INFO, report_) << "Creating inital map with " << map_->MapPointsInMap() << " map points";
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

void LocalMapper::clear()
{
    map_->clear();
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
        KeyFrame::Ptr keyframe = checkNewKeyFrame();
        if(keyframe)
        {
            mapTrace->startTimer("total");

            mapTrace->startTimer("dbow");
            finishNewKeyFrame(keyframe);
            mapTrace->stopTimer("dbow");

            std::list<MapPoint::Ptr> bad_mpts;
            int new_created_features = 0;
            int new_projected_features = 0;
            if(map_->kfs_.size() > 2)
            {
                mapTrace->startTimer("reproj");
                //new_created_features = createNewMapPoints(keyframe);
                //new_projected_features = searchInLocalMap(keyframe);
                mapTrace->stopTimer("reproj");
                LOG_IF(INFO, report_) << "[Mapper] create features: " << new_created_features << " + " << new_projected_features;

                mapTrace->startTimer("local_ba");
                Optimizer::localBundleAdjustment(keyframe, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
                mapTrace->stopTimer("local_ba");
            }

            for(const MapPoint::Ptr &mpt : bad_mpts)
            {
                map_->removeMapPoint(mpt);
            }

            checkCulling(keyframe);

            mapTrace->startTimer("dbow");
            addToDatabase(keyframe);
            mapTrace->stopTimer("dbow");

            mapTrace->stopTimer("total");
            mapTrace->writeToFile();

            keyframe_last_ = keyframe;
        }
    }
}

KeyFrame::Ptr LocalMapper::checkNewKeyFrame()
{
    std::unique_lock<std::mutex> lock(mutex_keyframe_);
    cond_process_.wait_for(lock, std::chrono::microseconds(5));

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
        return;

    map_->insertKeyFrame(keyframe);

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

        mapTrace->startTimer("dbow");
        finishNewKeyFrame(keyframe);
        mapTrace->stopTimer("dbow");

        std::list<MapPoint::Ptr> bad_mpts;
        int new_created_features = 0;
        int new_projected_features = 0;
        if(map_->kfs_.size() > 2)
        {
            mapTrace->startTimer("reproj");
            //new_created_features = createNewMapPoints(keyframe);
            //new_projected_features = searchInLocalMap(keyframe);
            mapTrace->stopTimer("reproj");
            LOG_IF(INFO, report_) << "[Mapper] create features: " << new_created_features << " + " << new_projected_features;

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

void LocalMapper::finishNewKeyFrame(const KeyFrame::Ptr &keyframe)
{
    keyframe->conputeDescriptor(brief_);
    keyframe->computeBoW(vocabulary_);

    const std::vector<size_t> indices = keyframe->getMapPointMatchIndices();
    for(const size_t &idx : indices)
    {
        const MapPoint::Ptr &mpt = keyframe->getMapPointByIndex(idx);
        mpt->computeDistinctiveDescriptor();
    }
}

void LocalMapper::createFeatureFromSeed(const Seed::Ptr &seed)
{
    //! create new feature
    if(seed->kf->getMapPointByIndex(seed->ft_idx))
        return;
    MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->getInvDepth()));
    seed->kf->removeSeedCreateByIndex(seed->ft_idx);
    seed->kf->addMapPoint(mpt, seed->ft_idx);
    map_->insertMapPoint(mpt);
    mpt->addObservation(seed->kf, seed->ft_idx);
    mpt->updateViewAndDepth();
    mpt->computeDistinctiveDescriptor();

//    if(mpt->observations() > 1)
//        Optimizer::refineMapPoint(mpt, 10, true);
}

void LocalMapper::createFeatureFromSeeds(const KeyFrame::Ptr &keyframe)
{
    return;

    const std::vector<size_t> seed_indices = keyframe->getSeedMatchIndices();
    const std::vector<Seed::Ptr> seeds = keyframe->getSeeds();

    for(const size_t &idx : seed_indices)
    {
        const Seed::Ptr &seed = seeds[idx];
        const KeyFrame::Ptr &keyframe_ref = seed->kf;
        const size_t &idx_ref = seed->ft_idx;

        const MapPoint::Ptr &mpt_ref = keyframe_ref->getMapPointByIndex(idx_ref);
        if(!mpt_ref)
        {
            MapPoint::Ptr mpt = MapPoint::create(keyframe_ref->Twc() * (seed->fn_ref/seed->getInvDepth()));
            keyframe_ref->removeSeedCreateByIndex(idx_ref);
            keyframe_ref->addMapPoint(mpt, idx_ref);
            keyframe->removeSeedMatchByIndex(idx);
            keyframe->addMapPoint(mpt, idx);

            map_->insertMapPoint(mpt);
            mpt->addObservation(keyframe_ref, idx_ref);
            mpt->addObservation(keyframe, idx);
            mpt->updateViewAndDepth();
            mpt->computeDistinctiveDescriptor();
        }
        else
        {

            const Vector3d pose = keyframe_ref->Tcw() * mpt_ref->pose();
            const double d_mpt = 1.0 / pose[2];

            const double sigma = std::sqrt(seed->getVariance());
            const double d_seed = seed->getInvDepth();
            if(std::abs(d_mpt-d_seed) <= sigma) //! add observation
            {
                keyframe->removeSeedMatchByIndex(idx);
                keyframe->addMapPoint(mpt_ref, idx);
                mpt_ref->addObservation(keyframe, idx);
                mpt_ref->updateViewAndDepth();
                mpt_ref->computeDistinctiveDescriptor();
            }
            else
            {
                keyframe->removeSeedMatchByIndex(idx);
            }
        }
    }
}

int LocalMapper::createNewMapPoints(const KeyFrame::Ptr &current_keyframe)
{
    static double focus_length = MIN(current_keyframe->cam_->fx(), current_keyframe->cam_->fy());
    static double pixel_usigma2 = 1.0/(focus_length*focus_length);
    static double epl_usigma2 = 3.841 * pixel_usigma2;

    std::set<KeyFrame::Ptr> connected_keyframes = current_keyframe->getConnectedKeyFrames(options_.num_reproject_kfs);

    const Vector3d Ow1 = current_keyframe->pose().translation();
    const Matrix4d Tcw1 = current_keyframe->Tcw().matrix();
    const Matrix3d Rcw1 = Tcw1.topLeftCorner<3,3>();
    const Vector3d tcw1 = Tcw1.topRightCorner<3,1>();
    const Matrix3d Rwc1 = Rcw1.transpose();

    const double ratio_factor = 1.5f*Frame::scale_factor_;
    int new_mpt_count = 0;
    for(const KeyFrame::Ptr &connected_keyframe : connected_keyframes)
    {
        const Vector3d Ow2 = connected_keyframe->pose().translation();
        const double baseline = (Ow1 - Ow2).norm();

        double depth_median, depth_min;
        connected_keyframe->getSceneDepth(depth_median, depth_min);

        const double ratio = baseline / depth_median;
        if(ratio < 0.01)
            continue;

        std::map<size_t, size_t> matches;
        FeatureTracker::searchBoWForTriangulation(current_keyframe, connected_keyframe, matches, 50, epl_usigma2);

//        std::vector<std::pair<size_t,size_t>> matches_vec(matches.begin(), matches.end());
//        FeatureTracker::showMatches(current_keyframe, connected_keyframe, matches_vec);

        const Matrix4d Tcw2 = connected_keyframe->Tcw().matrix();
        const Matrix3d Rcw2 = Tcw2.topLeftCorner<3,3>();
        const Vector3d tcw2 = Tcw2.topRightCorner<3,1>();
        const Matrix3d Rwc2 = Rcw2.transpose();
        for(const std::pair<size_t, size_t> &idx_pair : matches)
        {
            const size_t &idx1 = idx_pair.first;
            const size_t &idx2 = idx_pair.second;

            const Feature::Ptr &ft1 = current_keyframe->getFeatureByIndex(idx1);
            const Feature::Ptr &ft2 = connected_keyframe->getFeatureByIndex(idx2);

            //! check parallax
            const Vector3d ray1 = Rwc1 * ft1->fn_;
            const Vector3d ray2 = Rwc2 * ft2->fn_;
            const double cos_parallax = ray1.dot(ray2) / std::sqrt(ray1.squaredNorm() * ray2.squaredNorm());

            if(cos_parallax > 0.9998)// TODO how small the angle
                continue;

            //! triangulation
            MatrixXd A(4,4);
            A.row(0) = ft1->fn_[0]*Tcw1.row(2)-Tcw1.row(0);
            A.row(1) = ft1->fn_[1]*Tcw1.row(2)-Tcw1.row(1);
            A.row(2) = ft2->fn_[0]*Tcw2.row(2)-Tcw2.row(0);
            A.row(3) = ft2->fn_[1]*Tcw2.row(2)-Tcw2.row(1);

            JacobiSVD<MatrixXd> svd(A, ComputeThinV);
            MatrixXd V = svd.matrixV();

            Vector3d P3D = V.col(3).head<3>();
            P3D.array() /= V.col(3)[3];
            if(std::isinf(P3D[2]))
                continue;

            //! check reproject error
            double z1 = Rcw1.row(2).dot(P3D) + tcw1[2];
            if(z1 <= 0)
                continue;

            double z2 = Rcw2.row(2).dot(P3D) + tcw2[2];
            if(z2 <= 0)
                continue;

            const double& image1_sigma2 = Frame::level_sigma2_.at(ft1->corner_.level);
            const double x1 = Rcw1.row(0).dot(P3D) + tcw1[0];
            const double y1 = Rcw1.row(1).dot(P3D) + tcw1[1];

            const Vector2d rpj_px1 = current_keyframe->cam_->project(x1/z1, y1/z1);
            const Vector2d rpj_err1 = rpj_px1 - ft1->px_;
            const double rpj_err1_square = rpj_err1.squaredNorm();
            if(rpj_err1_square > image1_sigma2 * 5.991)
                continue;

            const double& image2_sigma2 = Frame::level_sigma2_.at(ft2->corner_.level);
            const double x2 = Rcw2.row(0).dot(P3D) + tcw2[0];
            const double y2 = Rcw2.row(1).dot(P3D) + tcw2[1];

            const Vector2d rpj_px2 = connected_keyframe->cam_->project(x2/z2, y2/z2);
            const Vector2d rpj_err2 = rpj_px2 - ft2->px_;
            const double rpj_err2_square = rpj_err2.squaredNorm();
            if(rpj_err2_square > image2_sigma2 * 5.991)
                continue;

            //! check scale consistency
            const Vector3d dist1 = P3D - Ow1;
            const double dist1_square = dist1.squaredNorm();

            const Vector3d dist2 = P3D - Ow2;
            const double dist2_square = dist2.squaredNorm();

            const double ratio_dist = std::sqrt(dist2_square/dist1_square);
            const double ratio_octave = Frame::scale_factors_[ft1->corner_.level]/Frame::scale_factors_[ft2->corner_.level];

            if(ratio_dist * ratio_factor < ratio_octave || ratio_octave * ratio_factor < ratio_dist)
                continue;

            if(0)
            {
                //! find subpixel feature
                Vector2d px2 = ft2->px_;
                bool succeed = FeatureTracker::findSubpixelFeature(current_keyframe, connected_keyframe, ft1, P3D, px2, ft2->corner_.level);
                if(!succeed)
                    continue;

                const Vector2d px_diff = ft2->px_ - px2;
                const double diff = px_diff.squaredNorm();

                if(diff > image2_sigma2 * 5.991)// TODO sigma2 should be?
                    continue;

                ft2->px_ = px2;
                ft2->fn_ = connected_keyframe->cam_->lift(px2);
                ft2->corner_.x = px2[0];
                ft2->corner_.y = px2[1];
            }

            MapPoint::Ptr new_mpt = MapPoint::create(P3D);
            map_->insertMapPoint(new_mpt);

            current_keyframe->removeSeedCreateByIndex(idx1);
            connected_keyframe->removeSeedCreateByIndex(idx2);

            current_keyframe->addMapPoint(new_mpt, idx1);
            connected_keyframe->addMapPoint(new_mpt, idx2);

            new_mpt->addObservation(current_keyframe, idx1);
            new_mpt->addObservation(connected_keyframe, idx2);
            new_mpt->updateViewAndDepth();
            new_mpt->computeDistinctiveDescriptor();

            new_mpt_count++;
        }
    }

    return new_mpt_count;
}

int LocalMapper::searchInLocalMap(const KeyFrame::Ptr &keyframe)
{
    const std::set<KeyFrame::Ptr> connected_keyframes = keyframe->getConnectedKeyFrames(options_.num_reproject_kfs);

    std::set<KeyFrame::Ptr> local_keyframes = connected_keyframes;
    if(connected_keyframes.size() < options_.num_reproject_kfs)
    {
        const std::set<KeyFrame::Ptr> sub_connected_keyframes = keyframe->getSubConnectedKeyFrames(options_.num_reproject_kfs - connected_keyframes.size());
        for(const KeyFrame::Ptr &kf : sub_connected_keyframes)
            local_keyframes.insert(kf);
    }

    std::unordered_set<MapPoint::Ptr> local_mpts_set;
    for(const KeyFrame::Ptr & kf : local_keyframes)
    {
        const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = kf->getMapPointFeaturesMatched();
        for(const auto &mpt_ft : mpt_fts)
        {
            if(!local_mpts_set.count(mpt_ft.first))
                local_mpts_set.insert(mpt_ft.first);
        }
    }

    std::vector<MapPoint::Ptr> local_mpts(local_mpts_set.begin(), local_mpts_set.end());
    std::map<MapPoint::Ptr, size_t> matches;
    FeatureTracker::searchBowByProjection(keyframe, local_mpts, matches, 50, 2.0);

    if(matches.empty())
        return 0;

    for(const auto & mpt_idx : matches)
    {
        const MapPoint::Ptr &mpt = mpt_idx.first;
        const size_t &idx = mpt_idx.second;

        const MapPoint::Ptr &mpt_kf = keyframe->getMapPointByIndex(idx);
        if(mpt_kf)
        {
            if(!mpt_kf->isBad())
            {
                mpt->fusion(mpt_kf);
                keyframe->removeMapPointMatchByIndex(idx);
                keyframe->addMapPoint(mpt, idx);
            }
            else
            {
                mpt_kf->removeObservation(keyframe);
                keyframe->removeMapPointMatchByIndex(idx);
                keyframe->addMapPoint(mpt, idx);
            }
        }
        else
        {
            keyframe->addMapPoint(mpt, idx);
            mpt->addObservation(keyframe, idx);
        }
    }

    std::vector<size_t > mpt_indices = keyframe->getMapPointMatchIndices();
    for(const size_t &idx : mpt_indices)
    {
        const MapPoint::Ptr &mpt = keyframe->getMapPointByIndex(idx);
        if(matches.count(mpt))
        {
            mpt->updateViewAndDepth();
            mpt->computeDistinctiveDescriptor();
        }
    }

    return (int)matches.size();
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

int LocalMapper::refineMapPoints(const int max_optimalize_num, const double outlier_thr)
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

        const std::map<KeyFrame::Ptr, size_t> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const size_t &idx = item.second;
            const Feature::Ptr &ft = kf->getFeatureByIndex(idx);
            double residual = utils::reprojectError(ft->fn_.head<2>()/ft->fn_[2], item.first->Tcw(), mpt->pose());
            if(residual < outlier_thr)
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
        std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        //std::cout << "mpt obs: [";
        for(const MapPoint::Ptr &mpt : mpts)
        {
            std::map<KeyFrame::Ptr, size_t> obs = mpt->getObservations();
            //std::cout << obs.size() << ",";
            if(obs.size() > observations_threshold)
            {
                int observations = 0;
//                const Feature::Ptr &ft = obs[kf];
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

//#ifdef SSVO_DBOW_ENABLE

//    std::vector<cv::KeyPoint> kps;
//    for(const Corner & corner : corners)
//    {
//        if(corner.x <= border_tl_[corner.level].x ||
//            corner.y <= border_tl_[corner.level].y ||
//            corner.x >= border_br_[corner.level].x ||
//            corner.y >= border_br_[corner.level].y)
//            continue;
//
//        kps.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));
//    }
//
//    BRIEF brief;
//    cv::Mat _descriptors;
//    brief.compute(frame->images(), kps, _descriptors);
//    std::vector<cv::Mat> descriptors;
//    descriptors.reserve(_descriptors.rows);
//    for(int i = 0; i < _descriptors.rows; i++)
//        descriptors.push_back(_descriptors.row(i));
//
//    DBoW3::BowVector bow_vec;
//    DBoW3::FeatureVector feat_vec;
//    vocabulary_.transform(descriptors, bow_vec, feat_vec, 4);
//
//    DBoW3::QueryResults results;
//    database_.query(bow_vec, results, 1);
//
//    if(results.empty())
//        return nullptr;
//
//    DBoW3::Result result = results[0];
//
//    reference = map_->getKeyFrame(result.Id);

//#endif

    // TODO 如果有关键帧剔除，则数据库索引存在问题。
    if(reference == nullptr)
        return nullptr;

    LOG_ASSERT(reference->dbow_Id_ == reference->id_) << "DBoW Id(" << reference->dbow_Id_ << ") is not match the keyframe's Id(" << reference->id_ << ")!";

    return reference;
}

}