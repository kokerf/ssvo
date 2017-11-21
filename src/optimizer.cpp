#include <iomanip>
#include "optimizer.hpp"

namespace ssvo{

Optimizer::Optimizer()
{
    options_.linear_solver_type = ceres::DENSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    //options_.max_solver_time_in_seconds = 0.2;
}

void Optimizer::report(bool with_residual)
{
    LOG(INFO) << summary_.FullReport();

    if(!with_residual)
        return;

    std::vector<ceres::ResidualBlockId> ids;
    problem_.GetResidualBlocks(&ids);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        LOG(INFO) << "BlockId: " << std::setw(5) << i <<" residual(RMSE): " << reprojectionError(problem_, ids[i]).norm();
    }
}

void Optimizer::twoViewBundleAdjustment(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, Map::Ptr map)
{
    ceres::LocalParameterization* local_parameterization = new ssvo::SE3Parameterization();

    kf1->optimal_Tw_ = kf1->pose();
    kf2->optimal_Tw_ = kf2->pose();

    problem_.AddParameterBlock(kf1->optimal_Tw_.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem_.AddParameterBlock(kf2->optimal_Tw_.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem_.SetParameterBlockConstant(kf1->optimal_Tw_.data());

    const Features &fts1 = kf1->fts_;
    MapPoints mpts;
    mpts.reserve(fts1.size());

    for(size_t id = 0; id < fts1.size();id++)
    {
        Feature::Ptr ft1 = fts1[id];
        MapPoint::Ptr mpt = ft1->mpt;
        if(mpt == nullptr)
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        if(ft2 == nullptr || ft2->mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        mpts.push_back(mpt);

        ceres::CostFunction* cost_function1 = ssvo::ReprojectionError::Create(ft1->ft[0], ft1->ft[1]);
        problem_.AddResidualBlock(cost_function1, NULL, kf1->optimal_Tw_.data(), mpt->optimal_pose_.data());

        ceres::CostFunction* cost_function2 = ssvo::ReprojectionError::Create(ft2->ft[0], ft2->ft[1]);
        problem_.AddResidualBlock(cost_function2, NULL, kf2->optimal_Tw_.data(), mpt->optimal_pose_.data());
    }

    ceres::Solve(options_, &problem_, &summary_);

    //! update pose
    kf2->setPose(kf2->optimal_Tw_);
    std::for_each(mpts.begin(), mpts.end(), [](MapPoint::Ptr mpt){mpt->setPose(mpt->optimal_pose_);});
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


}