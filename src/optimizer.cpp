#include <iomanip>
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace ssvo{


void Optimizer::twoViewBundleAdjustment(const KeyFrame::Ptr &kf1, const KeyFrame::Ptr &kf2, bool report, bool verbose)
{
    kf1->optimal_Tcw_ = kf1->Tcw();
    kf2->optimal_Tcw_ = kf2->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(kf1->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
    problem.AddParameterBlock(kf2->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(kf1->optimal_Tcw_.data());

    std::vector<Feature::Ptr> fts1;
    kf1->getFeatures(fts1);
    MapPoints mpts;

    for(const Feature::Ptr &ft1 : fts1)
    {
        MapPoint::Ptr mpt = ft1->mpt_;
        if(mpt == nullptr)//! should not happen
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        if(ft2 == nullptr || ft2->mpt_ == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        mpts.push_back(mpt);

        ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft1->fn_[0]/ft1->fn_[2], ft1->fn_[1]/ft1->fn_[2]);
        problem.AddResidualBlock(cost_function1, NULL, kf1->optimal_Tcw_.data(), mpt->optimal_pose_.data());

        ceres::CostFunction* cost_function2 = ceres_slover::ReprojectionErrorSE3::Create(ft2->fn_[0]/ft2->fn_[2], ft2->fn_[1]/ft2->fn_[2]);
        problem.AddResidualBlock(cost_function2, NULL, kf2->optimal_Tcw_.data(), mpt->optimal_pose_.data());
    }


    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
//    options_.gradient_tolerance = 1e-4;
//    options_.function_tolerance = 1e-4;
    //options_.max_solver_time_in_seconds = 0.2;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    kf2->setTcw(kf2->optimal_Tcw_);
    std::for_each(mpts.begin(), mpts.end(), [](MapPoint::Ptr mpt){mpt->setPose(mpt->optimal_pose_);});

    //! Report
    reportInfo(problem, summary, report, verbose);
}

void Optimizer::localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size, bool report, bool verbose)
{
    double t0 = (double)cv::getTickCount();
    std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size);
    actived_keyframes.insert(keyframe);
    std::unordered_set<MapPoint::Ptr> local_mappoints;
    std::list<KeyFrame::Ptr> fixed_keyframe;

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

            fixed_keyframe.push_back(item.first);
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

    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for(const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->setTcw(kf->optimal_Tcw_);
    }

    //! update mpts & remove mappoint with large error
    std::set<KeyFrame::Ptr> changed_keyframes;
    double max_residual = Config::pixelUnSigma2() * 2;
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
                         << ", KFs: " << actived_keyframes.size()
                         << ", Mpts: " << local_mappoints.size()
                         << ", remove " << bad_mpts.size() << " bad mpts."
                         << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";

    reportInfo(problem, summary, report, verbose);
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
//    reportInfo(problem, summary, report, verbose);
//}

void Optimizer::motionOnlyBundleAdjustment(const Frame::Ptr &frame, bool report, bool verbose)
{
    frame->optimal_Tcw_ = frame->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(frame->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);

    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    std::vector<Feature::Ptr> fts;
    frame->getFeatures(fts);
    for(const Feature::Ptr &ft : fts)
    {
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
        problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(mpt->optimal_pose_.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    frame->setTcw(frame->optimal_Tcw_);

    //! Report
    reportInfo(problem, summary, report, verbose);
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

//#define INVERSE_PARAMETERS
bool Optimizer::refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report, bool verbose)
{
    ceres::Problem problem;
    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    //! add obvers kf
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const KeyFrame::Ptr kf_ref = mpt->getReferenceKeyFrame();

#ifndef INVERSE_PARAMETERS
    mpt->optimal_pose_ = mpt->pose();
#endif

    for(const auto &obs_item : obs)
    {
        const KeyFrame::Ptr &kf = obs_item.first;
        const Feature::Ptr &ft = obs_item.second;
        kf->optimal_Tcw_ = kf->Tcw();

#ifdef INVERSE_PARAMETERS
        const Vector3d pose_ref = kf->Tcw() * mpt->pose();
        mpt->optimal_pose_ << pose_ref[0]/pose_ref[2], pose_ref[1]/pose_ref[2], 1.0/pose_ref[2];
        if(kf == kf_ref)
        {
            ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
            problem.AddParameterBlock(kf_ref->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
            problem.SetParameterBlockConstant(kf_ref->optimal_Tcw_.data());
            continue;
        }
        ceres::CostFunction *cost_function = ceres_slover::ReprojectionErrorSE3InvPoint::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
        problem.AddResidualBlock(cost_function, lossfunction, kf_ref->optimal_Tcw_.data(), kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
#else
        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
        problem.AddResidualBlock(cost_function, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
#endif
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = max_iter;

    ceres::Solve(options, &problem, &summary);

#ifdef INVERSE_PARAMETERS
    const Vector3d new_pose_ref(mpt->optimal_pose_[0]/mpt->optimal_pose_[2], mpt->optimal_pose_[1]/mpt->optimal_pose_[2], 1.0/mpt->optimal_pose_[2]);
    mpt->setPose(kf_ref->Twc() * new_pose_ref);
#else
    mpt->setPose(mpt->optimal_pose_);
#endif

    std::set<KeyFrame::Ptr> changed_keyframes;
    double max_residual = Config::pixelUnSigma2() * 2;

    bool isGood = true;
    for(const auto &item : obs)
    {
        double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->pose());
        if(residual < max_residual)
            continue;

        mpt->removeObservation(item.first);
        changed_keyframes.insert(item.first);

        if(mpt->type() == MapPoint::BAD)
        {
            isGood = false;
        }
    }

    for(const KeyFrame::Ptr &kf : changed_keyframes)
    {
        kf->updateConnections();
    }

    reportInfo(problem, summary, report, verbose);
    return isGood;
}

Vector2d Optimizer::reprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id)
{
    auto cost = problem.GetCostFunctionForResidualBlock(id);
    std::vector<double*> parameterBlocks;
    problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
    Vector2d residual;
    cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
    return residual;
}

void Optimizer::reportInfo(const ceres::Problem &problem, const ceres::Solver::Summary summary, bool report, bool verbose)
{
    if(!report) return;

    if(!verbose)
    {
        LOG(INFO) << summary.BriefReport();
    }
    else
    {
        LOG(INFO) << summary.FullReport();
        std::vector<ceres::ResidualBlockId> ids;
        problem.GetResidualBlocks(&ids);
        for (size_t i = 0; i < ids.size(); ++i)
        {
            LOG(INFO) << "BlockId: " << std::setw(5) << i <<" residual(RMSE): " << reprojectionError(problem, ids[i]).norm();
        }
    }
}


}