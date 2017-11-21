#include "bundle_adjustment.hpp"

namespace ssvo{

namespace BA {

void setOptions(ceres::Solver::Options& options, bool output)
{
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = output;
    //options.max_solver_time_in_seconds = 0.2;
}

void sloveReport(ceres::Solver::Summary& summary, bool output)
{
    if(output) LOG(INFO) << summary.FullReport();
}

void twoViewBA(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, Map::Ptr map)
{
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::SE3Parameterization();

    KeyFramePose kf1_pose(kf1, kf1->pose());
    KeyFramePose kf2_pose(kf2, kf2->pose());

    problem.AddParameterBlock(kf1_pose.se3.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem.AddParameterBlock(kf2_pose.se3.data(), Sophus::SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(kf1_pose.se3.data());

    const Features &fts1 = kf1->fts_;
    std::vector<MapPointPose> mpts_pose;
    mpts_pose.reserve(fts1.size());

    for(size_t id = 0; id < fts1.size();id++)
    {
        Feature::Ptr ft1 = fts1[id];
        MapPoint::Ptr mpt = ft1->mpt;
        if(mpt == nullptr)
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        if(ft2 == nullptr || ft2->mpt == nullptr)
            continue;

        mpts_pose.push_back(MapPointPose(mpt, mpt->pose()));

        ceres::CostFunction* cost_function1 = ssvo::ReprojectionError::Create(ft1->ft[0], ft1->ft[1]);
        problem.AddResidualBlock(cost_function1, NULL, kf1_pose.se3.data(), mpts_pose.back().t);

        ceres::CostFunction* cost_function2 = ssvo::ReprojectionError::Create(ft2->ft[0], ft2->ft[1]);
        problem.AddResidualBlock(cost_function2, NULL, kf2_pose.se3.data(), mpts_pose.back().t);
    }

    ceres::Solver::Options options;
    setOptions(options, true);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //! update pose
    kf2_pose.update();

    std::vector<MapPointPose>::iterator mpts_pose_iter = mpts_pose.begin();
    std::vector<MapPointPose>::iterator mpts_pose_end = mpts_pose.end();
    for(; mpts_pose_iter != mpts_pose_end ; ++mpts_pose_iter)
    {
        mpts_pose_iter->update();
    }

    std::vector<ceres::ResidualBlockId> ids;
    problem.GetResidualBlocks(&ids);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        LOG(INFO) << i <<"Error:" << reprojectionError(problem, ids[i]).norm();
    }

    sloveReport(summary, true);
}

Vector2d reprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id)
{
    auto cost = problem.GetCostFunctionForResidualBlock(id);
    std::vector<double*> parameterBlocks;
    problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
    Vector2d residual;
    cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
    return residual;
}

}


}