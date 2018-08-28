#include <iomanip>
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace ssvo{


void Optimizer::globleBundleAdjustment(const Map::Ptr &map, int max_iters, bool report, bool verbose)
{
    if (map->KeyFramesInMap() < 2)
        return;

    std::vector<KeyFrame::Ptr> all_kfs = map->getAllKeyFrames();
    std::vector<MapPoint::Ptr> all_mpts = map->getAllMapPoints();

    static double focus_length = MIN(all_kfs.back()->cam_->fx(), all_kfs.back()->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma() / focus_length;

    ceres::Problem problem;

    for (const KeyFrame::Ptr &kf : all_kfs)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        if(kf->id_ == 0)
            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    double scale = pixel_usigma * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
    for (const MapPoint::Ptr &mpt : all_mpts)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for (const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_num_iterations = max_iters;
//    options_.gradient_tolerance = 1e-4;
//    options_.function_tolerance = 1e-4;
    //options_.max_solver_time_in_seconds = 0.2;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    std::for_each(all_kfs.begin(), all_kfs.end(), [](KeyFrame::Ptr kf) {kf->setTcw(kf->optimal_Tcw_); });
    std::for_each(all_mpts.begin(), all_mpts.end(), [](MapPoint::Ptr mpt){mpt->setPose(mpt->optimal_pose_);});

    //! Report
    reportInfo<2>(problem, summary, report, verbose);
}

void Optimizer::localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int max_iters, int size, int min_shared_fts, bool report, bool verbose)
{
    static double focus_length = MIN(keyframe->cam_->fx(), keyframe->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma()/focus_length;

    double t0 = (double)cv::getTickCount();
    size = size > 0 ? size-1 : 0;
    std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size, min_shared_fts);
    actived_keyframes.insert(keyframe);
    std::unordered_set<MapPoint::Ptr> local_mappoints;
    std::set<KeyFrame::Ptr> fixed_keyframe;

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        MapPoints mpts;
        kf->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            local_mappoints.insert(mpt);
        }
    }

    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            if(actived_keyframes.count(item.first))
                continue;

            fixed_keyframe.insert(item.first);
        }
    }

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();

    for(const KeyFrame::Ptr &kf : fixed_keyframe)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        if(kf->id_ <= 1)
            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    double scale = pixel_usigma * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for(const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_num_iterations = max_iters;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->setTcw(kf->optimal_Tcw_);
    }

    //! update mpts & remove mappoint with large error
    std::set<KeyFrame::Ptr> changed_keyframes;
    static const double max_residual = pixel_usigma * pixel_usigma * std::sqrt(3.81);
    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
            if(residual < max_residual)
                continue;

            mpt->removeObservation(item.first);
            changed_keyframes.insert(item.first);
//            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;

            if(mpt->type() == MapPoint::BAD)
            {
                bad_mpts.push_back(mpt);
            }
        }

        mpt->setPose(mpt->optimal_pose_);
    }

    for(const KeyFrame::Ptr &kf : changed_keyframes)
    {
        kf->updateConnections();
    }

    //! Report
    double t1 = (double)cv::getTickCount();
    LOG_IF(INFO, report) << "[Optimizer] Finish local BA for KF: " << keyframe->id_ << "(" << keyframe->frame_id_ << ")"
                         << ", KFs: " << actived_keyframes.size() << "(+" << fixed_keyframe.size() << ")"
                         << ", Mpts: " << local_mappoints.size()
                         << ", remove " << bad_mpts.size() << " bad mpts."
                         << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";

    reportInfo<2>(problem, summary, report, verbose);
}

//void Optimizer::localBundleAdjustmentWithInvDepth(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size, bool report, bool verbose)
//{
//    double t0 = (double)cv::getTickCount();
//    std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size);
//    actived_keyframes.insert(keyframe);
//    std::unordered_set<MapPoint::Ptr> local_mappoints;
//    std::list<KeyFrame::Ptr> fixed_keyframe;
//
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        MapPoints mpts;
//        kf->getMapPoints(mpts);
//        for(const MapPoint::Ptr &mpt : mpts)
//        {
//            local_mappoints.insert(mpt);
//        }
//    }
//
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//        for(const auto &item : obs)
//        {
//            if(actived_keyframes.count(item.first))
//                continue;
//
//            fixed_keyframe.push_back(item.first);
//        }
//    }
//
//    ceres::Problem problem;
//    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
//
//    std::unordered_set<KeyFrame::Ptr> local_keyframes;
//    for(const KeyFrame::Ptr &kf : fixed_keyframe)
//    {
//        local_keyframes.insert(kf);
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }
//
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        local_keyframes.insert(kf);
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        if(kf->id_ <= 1)
//            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }
//
//    double scale = Config::pixelUnSigma() * 2;
//    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
//    std::unordered_map<MapPoint::Ptr, KeyFrame::Ptr> optimal_invdepth_mappoints;
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        KeyFrame::Ptr ref_kf = mpt->getReferenceKeyFrame();
//        if(local_keyframes.count(ref_kf))
//        {
//            optimal_invdepth_mappoints.emplace(mpt, ref_kf);
//            Vector3d pose = ref_kf->Tcw() * mpt->pose();
//            mpt->optimal_pose_ << pose[0]/pose[2], pose[1]/pose[2], 1.0/pose[2];
//            const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//            const Feature::Ptr ref_ft = obs.find(ref_kf)->second;
//            for(const auto &item : obs)
//            {
//                const KeyFrame::Ptr &kf = item.first;
//                if(kf == ref_kf)
//                    continue;
//
//                const Feature::Ptr &ft = item.second;
//                ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3InvDepth::Create(
//                    ref_ft->fn_[0]/ref_ft->fn_[2], ref_ft->fn_[1]/ref_ft->fn_[2],
//                    ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
//                problem.AddResidualBlock(cost_function1, lossfunction, ref_kf->optimal_Tcw_.data(), kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
//            }
//        }
//        else
//        {
//            mpt->optimal_pose_ = mpt->pose();
//            const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//            for(const auto &item : obs)
//            {
//                const KeyFrame::Ptr &kf = item.first;
//                const Feature::Ptr &ft = item.second;
//                ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
//                problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
//            }
//        }
//    }
//
//    ceres::Solver::Options options;
//    ceres::Solver::Summary summary;
//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;//report & verbose;
//
//    ceres::Solve(options, &problem, &summary);
//
//    //! update pose
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        kf->setTcw(kf->optimal_Tcw_);
//    }
//
//    //! update mpts & remove mappoint with large error
//    std::set<KeyFrame::Ptr> changed_keyframes;
//    double max_residual = Config::pixelUnSigma2() * 2;
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        if(optimal_invdepth_mappoints.count(mpt))
//        {
//            KeyFrame::Ptr ref_kf = optimal_invdepth_mappoints[mpt];
//            Feature::Ptr ref_ft = ref_kf->getFeatureByMapPoint(mpt);
//            mpt->optimal_pose_ << mpt->optimal_pose_[0]/mpt->optimal_pose_[2], mpt->optimal_pose_[1]/mpt->optimal_pose_[2], 1.0/mpt->optimal_pose_[2];
//            mpt->optimal_pose_ = ref_kf->Twc() * mpt->optimal_pose_;
//        }
//
//        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//        for(const auto &item : obs)
//        {
//            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
//            if(residual < max_residual)
//                continue;
//
//            mpt->removeObservation(item.first);
//            changed_keyframes.insert(item.first);
////            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;
//
//            if(mpt->type() == MapPoint::BAD)
//            {
//                bad_mpts.push_back(mpt);
//            }
//        }
//
//        mpt->setPose(mpt->optimal_pose_);
//    }
//
//    for(const KeyFrame::Ptr &kf : changed_keyframes)
//    {
//        kf->updateConnections();
//    }
//
//    //! Report
//    double t1 = (double)cv::getTickCount();
//    LOG_IF(INFO, report) << "[Optimizer] Finish local BA for KF: " << keyframe->id_ << "(" << keyframe->frame_id_ << ")"
//                         << ", KFs: " << actived_keyframes.size()
//                         << ", Mpts: " << local_mappoints.size() << "(" << optimal_invdepth_mappoints.size() << ")"
//                         << ", remove " << bad_mpts.size() << " bad mpts."
//                         << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";
//
//    reportInfo<2>(problem, summary, report, verbose);
//}

void Optimizer::motionOnlyBundleAdjustment(const Frame::Ptr &frame, bool use_seeds, bool reject, bool report, bool verbose)
{
    const double focus_length = MIN(frame->cam_->fx(), frame->cam_->fy());
    const double pixel_usigma = Config::imagePixelSigma()/focus_length;

    static const size_t OPTIMAL_MPTS = 150;

    frame->optimal_Tcw_ = frame->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(frame->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);

    static const double scale = pixel_usigma * std::sqrt(3.81);
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    std::vector<Feature::Ptr> fts;
    frame->getFeatures(fts);
    const size_t N = fts.size();
    std::vector<ceres::ResidualBlockId> res_ids(N);
    for(size_t i = 0; i < N; ++i)
    {
        Feature::Ptr ft = fts[i];
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
        res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(mpt->optimal_pose_.data());
    }

    if(N < OPTIMAL_MPTS)
    {
        std::vector<Feature::Ptr> ft_seeds;
        frame->getSeeds(ft_seeds);
        const size_t needed = OPTIMAL_MPTS - N;
        if(ft_seeds.size() > needed)
        {
            std::nth_element(ft_seeds.begin(), ft_seeds.begin()+needed, ft_seeds.end(),
                             [](const Feature::Ptr &a, const Feature::Ptr &b)
                             {
                               return a->seed_->getInfoWeight() > b->seed_->getInfoWeight();
                             });

            ft_seeds.resize(needed);
        }

        const size_t M = ft_seeds.size();
        res_ids.resize(N+M);
        for(int i = 0; i < M; ++i)
        {
            Feature::Ptr ft = ft_seeds[i];
            Seed::Ptr seed = ft->seed_;
            if(seed == nullptr)
                continue;

            seed->optimal_pose_.noalias() = seed->kf->Twc() * (seed->fn_ref / seed->getInvDepth());

            ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(seed->fn_ref[0]/seed->fn_ref[2], seed->fn_ref[1]/seed->fn_ref[2], seed->getInfoWeight());
            res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), seed->optimal_pose_.data());
            problem.SetParameterBlockConstant(seed->optimal_pose_.data());

        }
    }


    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    if(reject)
    {
        int remove_count = 0;

        static const double TH_REPJ = 3.81 * pixel_usigma * pixel_usigma;
        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];
            if(evaluateResidual<2>(problem, res_ids[i]).squaredNorm() > TH_REPJ * (1 << ft->level_))
            {
                remove_count++;
                problem.RemoveResidualBlock(res_ids[i]);
                frame->removeFeature(ft);
            }
        }

        ceres::Solve(options, &problem, &summary);

        LOG_IF(WARNING, report) << "[Optimizer] Motion-only BA removes " << remove_count << " points";
    }

    //! update pose
    frame->setTcw(frame->optimal_Tcw_);

    //! Report
    reportInfo<2>(problem, summary, report, verbose);
}

void Optimizer::refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report, bool verbose)
{

#if 0
    ceres::Problem problem;
    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    //! add obvers kf
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const KeyFrame::Ptr kf_ref = mpt->getReferenceKeyFrame();

    mpt->optimal_pose_ = mpt->pose();

    for(const auto &obs_item : obs)
    {
        const KeyFrame::Ptr &kf = obs_item.first;
        const Feature::Ptr &ft = obs_item.second;
        kf->optimal_Tcw_ = kf->Tcw();

        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
        problem.AddResidualBlock(cost_function, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = max_iter;

    ceres::Solve(options, &problem, &summary);

    mpt->setPose(mpt->optimal_pose_);

    reportInfo<2>(problem, summary, report, verbose);
#else

    double t0 = (double)cv::getTickCount();
    mpt->optimal_pose_ = mpt->pose();
    Vector3d pose_last = mpt->optimal_pose_;
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const size_t n_obs = obs.size();

    Matrix3d A;
    Vector3d b;
    double init_chi2 = std::numeric_limits<double>::max();
    double last_chi2 = std::numeric_limits<double>::max();
    const double EPS = 1E-10;

    const bool progress_out = report&verbose;
    bool convergence = false;
    int i = 0;
    for(; i < max_iter; ++i)
    {
        A.setZero();
        b.setZero();
        double new_chi2 = 0.0;

        //! compute res
        for(const auto &obs_item : obs)
        {
            const SE3d Tcw = obs_item.first->Tcw();
            const Vector2d fn = obs_item.second->fn_.head<2>();

            const Vector3d point(Tcw * mpt->optimal_pose_);
            const Vector2d resduial(point.head<2>()/point[2] - fn);

            new_chi2 += resduial.squaredNorm();

            Eigen::Matrix<double, 2, 3> Jacobain;

            const double z_inv = 1.0 / point[2];
            const double z_inv2 = z_inv*z_inv;
            Jacobain << z_inv, 0.0, -point[0]*z_inv2, 0.0, z_inv, -point[1]*z_inv2;

            Jacobain = Jacobain * Tcw.rotationMatrix();

            A.noalias() += Jacobain.transpose() * Jacobain;
            b.noalias() -= Jacobain.transpose() * resduial;
        }

        if(i == 0)  {init_chi2 = new_chi2;}

        if(last_chi2 < new_chi2)
        {
            LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": failure, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs;
            mpt->setPose(pose_last);
            return;
        }

        last_chi2 = new_chi2;

        const Vector3d dp(A.ldlt().solve(b));

        pose_last = mpt->optimal_pose_;
        mpt->optimal_pose_.noalias() += dp;

        LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": success, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs << ", step: " << dp.transpose();

        if(dp.norm() <= EPS)
        {
            convergence = true;
            break;
        }
    }

    mpt->setPose(mpt->optimal_pose_);
    double t1 = (double)cv::getTickCount();
    LOG_IF(INFO, verbose) << std::scientific  << "[Optimizer] MapPoint " << mpt->id_
                         << " Error(MSE) changed from " << std::scientific << init_chi2/n_obs << " to " << last_chi2/n_obs
                         << "(" << obs.size() << "), time: " << std::fixed << (t1-t0)*1000/cv::getTickFrequency() << "ms, "
                         << (convergence? "Convergence" : "Unconvergence");

#endif
}


//! ===============================  for vio  ====================================
bool Optimizer::sloveInitialGyroBias(const std::vector<Frame::Ptr> &frames, Vector3d &dbias_gyro, bool report, bool verbose)
{
    dbias_gyro.setZero();
    const size_t N = frames.size();
    if (N < 2)
        return false;

    ceres::Problem problem;
    dbias_gyro = Vector3d(0.0, 0.0, 0.0);

    std::vector<ceres::ResidualBlockId> res_ids(N - 1);
    for (size_t i = 0; i < N; i++)
    {
        if (0 == i)
        {
            Frame::Ptr framei = frames[i];
            framei->optimal_Twb_ = framei->Twc() * SE3d(framei->cam_->T_CB());
            ceres::LocalParameterization* local_parameterization_ri = new ceres_slover::SO3Parameterization();
            problem.AddParameterBlock(frames[0]->optimal_Twb_.data(), SO3d::num_parameters, local_parameterization_ri);
            problem.SetParameterBlockConstant(frames[0]->optimal_Twb_.data());
            continue;
        }

        Frame::Ptr framei = frames[i - 1];
        Frame::Ptr framej = frames[i];

        framej->optimal_Twb_ = framej->Twc() * SE3d(framej->cam_->T_CB());
        ceres::LocalParameterization* local_parameterization_rj = new ceres_slover::SO3Parameterization();
        problem.AddParameterBlock(framej->optimal_Twb_.data(), SO3d::num_parameters, local_parameterization_rj);
        problem.SetParameterBlockConstant(framej->optimal_Twb_.data());

        ceres::CostFunction* cost_function = ceres_slover::PreintegrationRotationError::Create(&framej->getPreintergrationConst());
        res_ids[i-1] = problem.AddResidualBlock(cost_function, nullptr, framei->optimal_Twb_.data(), framej->optimal_Twb_.data(), dbias_gyro.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    reportInfo<3>(problem, summary, report, verbose);

    return true;

}

bool Optimizer::sloveScaleAndGravity(const std::vector<Frame::Ptr> &frames, Vector4d &scale_and_gravity, double threshold, bool verbose)
{
    scale_and_gravity.setZero();
    const size_t N = frames.size();
    if (N < 4)
        return false;

    const size_t M = N - 2;

    MatrixXd A; A.resize(3 * M, 4);
    VectorXd b; b.resize(3 * M);

    const Matrix4d Tcb = frames.back()->cam_->T_CB();
    const Matrix3d Rcb = Tcb.topLeftCorner<3, 3>();
    const Vector3d tcb = Tcb.topRightCorner<3, 1>();
    const Matrix3d half_I3x3 = 0.5 * Matrix3d::Identity(3, 3);
    for (size_t i = 0; i < M; i++)
    {
        const Frame::Ptr frame1 = frames[i];
        const Frame::Ptr frame2 = frames[i+1];
        const Frame::Ptr frame3 = frames[i+2];

        const Preintegration & preint12 = frame2->getPreintergrationConst();
        const Preintegration & preint23 = frame3->getPreintergrationConst();
        const Vector3d & dv12 = preint12.deltaVij();
        const Vector3d & dp12 = preint12.deltaPij();
        const Vector3d & dp23 = preint23.deltaPij();

        const double dt12 = preint12.deltaTij();
        const double dt23 = preint23.deltaTij();
        const double dt12dt23 = dt12 * dt23;

        const Vector3d pwc1 = frame1->Twc().translation();
        const Vector3d pwc2 = frame2->Twc().translation();
        const Vector3d pwc3 = frame3->Twc().translation();
        const Matrix3d Rwc1 = frame1->Twc().rotationMatrix();
        const Matrix3d Rwc2 = frame2->Twc().rotationMatrix();
        const Matrix3d Rwc3 = frame3->Twc().rotationMatrix();

        //! lambda
        A.block<3, 1>(3 * i, 0) = (pwc2 - pwc3)*dt12 + (pwc2 - pwc1)*dt23;
        //! beta
        A.block<3, 3>(3 * i, 1) = dt12dt23 * (dt12 + dt23) * half_I3x3;
        //! gamma
        b.segment<3>(3 * i) = Rwc1 * Rcb * (dp12 * dt23 - dv12 * dt12dt23) - Rwc2 * Rcb * dp23 * dt12 
            + (Rwc3 - Rwc2) * tcb * dt12 + (Rwc1 - Rwc2) * tcb * dt23;
    }

#if EIGEN_VERSION_AT_LEAST(3,3,0)
    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
#else
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
#endif

    Vector4d x = svd.solve(b);
    scale_and_gravity = x;

    Vector4d sigular_values = svd.singularValues();
    const double max_sigular_value = sigular_values.maxCoeff();
    const double min_sigular_value = sigular_values.minCoeff();
    const double condition_number = max_sigular_value / min_sigular_value;
    LOG_IF(INFO, verbose) << " Slove Scale & Graity, x: " << x.transpose() <<", sigular value: " << sigular_values.transpose() << ", cond: " << condition_number;

    if (min_sigular_value < 1e-10 || condition_number > threshold || scale_and_gravity[0] <= 0.0)
        return false;

    return true;
}

bool Optimizer::sloveInitialAccBiasAndRefine(const std::vector<Frame::Ptr> &frames, Vector4d &scale_and_gravity, Vector3d &dbias_acc, double threshold, bool verbose)
{
    dbias_acc.setZero();
    const size_t N = frames.size();
    if (N < 4)
        return false;

    //! slove
    const size_t M = N - 2;

    MatrixXd A; A.resize(3 * M, 6);
    VectorXd b; b.resize(3 * M);

    const Matrix4d Tcb = frames.back()->cam_->T_CB();
    const Matrix3d Rcb = Tcb.topLeftCorner<3, 3>();
    const Vector3d tcb = Tcb.topRightCorner<3, 1>();
    const Matrix3d negtive_half_I3x3 = -0.5 * Matrix3d::Identity(3, 3);

    const Vector3d gravity_nominal(0.0, 0.0, -1.0);
    const Vector3d gravity_estimate0 = scale_and_gravity.tail<3>();
    const double G = IMUPara::gravity();

    double scale = 1.0;
    Vector3d gravity_estimate = gravity_estimate0;

    for (size_t iter = 0; iter < 4; iter++)
    {
        //! gI vs gW, in direction
        const Vector3d gravity_world = gravity_estimate.normalized();

        const Vector3d gI_cross_gW = SO3d::hat(gravity_nominal) * gravity_world;
        const double gI_dot_gW = gravity_nominal.dot(gravity_world);
        const double gI_cross_gW_norm = gI_cross_gW.norm();
        const double theta = std::atan2(gI_cross_gW_norm, gI_dot_gW);
        SO3d Rwi = SO3d::exp((theta / gI_cross_gW_norm)* gI_cross_gW);

        const Vector3d gW0 = Rwi * gravity_nominal * G;
        const Matrix3d negtive_half_Rwi_hatgI = -0.5 * Rwi.matrix() * SO3d::hat(gravity_nominal) * G;


        for (size_t i = 0; i < M; i++)
        {
            const Frame::Ptr frame1 = frames[i];
            const Frame::Ptr frame2 = frames[i + 1];
            const Frame::Ptr frame3 = frames[i + 2];

            const Preintegration & preint12 = frame2->getPreintergrationConst();
            const Preintegration & preint23 = frame3->getPreintergrationConst();
            const Vector3d & dv12 = preint12.deltaVij();
            const Vector3d & dp12 = preint12.deltaPij();
            const Vector3d & dp23 = preint23.deltaPij();
            const Matrix3d & Jdpba12 = preint12.jacobdPBiasAcc();
            const Matrix3d & Jdpba23 = preint23.jacobdPBiasAcc();
            const Matrix3d & Jdvba12 = preint12.jacobdVBiasAcc();

            const double dt12 = preint12.deltaTij();
            const double dt23 = preint23.deltaTij();
            const double dt12dt23 = dt12 * dt23;
            const double dt12dt23_sum = dt12dt23 * (dt12 + dt23);

            const Vector3d pwc1 = frame1->Twc().translation();
            const Vector3d pwc2 = frame2->Twc().translation();
            const Vector3d pwc3 = frame3->Twc().translation();
            const Matrix3d Rwc1 = frame1->Twc().rotationMatrix();
            const Matrix3d Rwc2 = frame2->Twc().rotationMatrix();
            const Matrix3d Rwc3 = frame3->Twc().rotationMatrix();

            //! lambda
            A.block<3, 1>(3 * i, 0) = (pwc2 - pwc3)*dt12 + (pwc2 - pwc1)*dt23;
            //! phi
            A.block<3, 2>(3 * i, 1) = dt12dt23_sum * negtive_half_Rwi_hatgI.block<3, 2>(0, 0);
            //! zeta
            A.block<3, 3>(3 * i, 3) = Rwc1 * Rcb * dt23 * (Jdvba12 * dt12 - Jdpba12) + Rwc2 * Rcb * Jdpba23 * dt12;
            //! psi
            b.segment<3>(3 * i) = Rwc1 * Rcb * (dp12 * dt23 - dv12 * dt12dt23) - Rwc2 * Rcb * dp23 * dt12
                + (Rwc3 - Rwc2) * tcb * dt12 + (Rwc1 - Rwc2) * tcb * dt23
                - 0.5 * dt12dt23_sum * gW0;
        }

#if EIGEN_VERSION_AT_LEAST(3,3,0)
        BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
#else
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
#endif
        VectorXd x = svd.solve(b);

        const Vector3d delta_theta(x[1], x[2], 0);
        gravity_estimate = Rwi * SO3d::exp(delta_theta) * gravity_nominal * G;
        scale = x[0];
        dbias_acc = x.tail<3>();
        scale_and_gravity[0] = scale;
        scale_and_gravity.tail<3>() = gravity_estimate;

        VectorXd sigular_values = svd.singularValues();
        const double max_sigular_value = sigular_values.maxCoeff();
        const double min_sigular_value = sigular_values.minCoeff();
        const double condition_number = max_sigular_value / min_sigular_value;
        bool well_confitioned = min_sigular_value > 1e-10 && condition_number < threshold;

        LOG_IF(INFO, verbose) << " Iter: " << iter << ", x: " << x.transpose() << ", g: " << gravity_estimate.transpose() << ", cond: " << well_confitioned << "(" << sigular_values.transpose() << ")";

        if (!well_confitioned || scale <= 0)
            return false;

        if (delta_theta.head<2>().norm() < 1e-10)
            break;
    }

    static const double min_rot_angle = 3.1415926 / 6;//! for big acc bias, may be larger?
    const double rot_angle = std::acos(gravity_estimate.normalized().transpose() * gravity_estimate0.normalized());
    if (rot_angle > min_rot_angle)
        return false;

    return true;
}

bool Optimizer::initIMU(const std::vector<Frame::Ptr> &frames, VectorXd &result, bool report, bool verbose)
{
    result.setZero();
    const size_t N = frames.size();
    if (N < 2)
        return false;

    bool succeed = false;

    //! check frame order
    for (size_t i = 1; i < N; i++)
    {
        const Frame::Ptr &framei = frames[i - 1];
        const Frame::Ptr &framej = frames[i];

        LOG_ASSERT(std::abs(framei->timestamp_ - framej->getPreintergrationConst().Ti()) < 1e-4)
            << "Error frame data for imu init! Ti: " << framej->getPreintergrationConst().Ti()
            << ", Fi: " << framei->id_ << "(" << framei->timestamp_ << "),"
            << ", Fj: " << framej->id_ << "(" << framej->timestamp_ << ")";
    }

    //! slove gyro bias
    Vector3d dbias_gyro;
    succeed = Optimizer::sloveInitialGyroBias(frames, dbias_gyro, report, verbose);
    result = dbias_gyro;
    if (!succeed) return false;

    LOG_IF(INFO, report) << "[Optimizer] [1/3] dbias_gyro: " << dbias_gyro.transpose() << std::endl;

    //! slove scale and gravity
    Vector4d scale_and_gravity;
    succeed = Optimizer::sloveScaleAndGravity(frames, scale_and_gravity, 1e3, verbose);
    result.resize(7);
    result.head<3>() = dbias_gyro;
    result.tail<4>() = scale_and_gravity;
    if (!succeed) return false;

    LOG_IF(INFO, report) << "[Optimizer] [2/3] scale_and_gravity: " << scale_and_gravity.transpose() << std::endl;

    //! slove acc bias and refine
    Vector4d scale_and_gravity_new = scale_and_gravity;
    Vector3d dbias_acc;
    succeed = Optimizer::sloveInitialAccBiasAndRefine(frames, scale_and_gravity_new, dbias_acc, 1e3, verbose);
    result.resize(10);
    result.head<3>() = dbias_gyro;
    result.segment<4>(3) = scale_and_gravity_new;
    result.tail<3>() = dbias_acc;
    if (!succeed) return false;

    LOG_IF(INFO, report) << "[Optimizer] [3/3] scale_and_gravity: " << scale_and_gravity_new.transpose() << ", delta_bias_acc: " << dbias_acc.transpose() << std::endl;

    return true;
}

}