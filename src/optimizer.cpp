#include <iomanip>
#include "optimizer.hpp"
#include "config.hpp"

namespace ssvo{


void Optimizer::twoViewBundleAdjustment(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, bool report, bool verbose)
{
    kf1->optimal_Tcw_ = kf1->Tcw();
    kf2->optimal_Tcw_ = kf2->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(kf1->optimal_Tcw_.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem.AddParameterBlock(kf2->optimal_Tcw_.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(kf1->optimal_Tcw_.data());

    const std::vector<Feature::Ptr> fts1 = kf1->getFeatures();
    MapPoints mpts;
    mpts.reserve(fts1.size());

    for(Feature::Ptr ft1 : fts1)
    {
        MapPoint::Ptr mpt = ft1->mpt;
        if(mpt == nullptr)//! should not happen
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        if(ft2 == nullptr || ft2->mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        mpts.push_back(mpt);

        ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft1->fn[0]/ft1->fn[2], ft1->fn[1]/ft1->fn[2]);
        problem.AddResidualBlock(cost_function1, NULL, kf1->optimal_Tcw_.data(), mpt->optimal_pose_.data());

        ceres::CostFunction* cost_function2 = ceres_slover::ReprojectionErrorSE3::Create(ft2->fn[0]/ft2->fn[2], ft2->fn[1]/ft2->fn[2]);
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

void Optimizer::motionOnlyBundleAdjustment(Frame::Ptr frame, bool report, bool verbose)
{
    frame->optimal_Tcw_ = frame->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(frame->optimal_Tcw_.data(), Sophus::SE3d::num_parameters, local_parameterization);

    double scale = Config::pixelUnSigma2() * 4;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    std::vector<Feature::Ptr> fts = frame->getFeatures();
    for(Feature::Ptr ft : fts)
    {
        MapPoint::Ptr mpt = ft->mpt;
        if(mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn[0]/ft->fn[2], ft->fn[1]/ft->fn[2]);
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