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
    if(output) std::cout << summary.FullReport() << std::endl;
}

bool twoViewBA(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, Map::Ptr map)
{
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

    KeyFramePose kf1_pose(kf1, kf1->getRotation(), kf1->getTranslation());
    KeyFramePose kf2_pose(kf2, kf2->getRotation(), kf2->getTranslation());

    problem.AddParameterBlock(kf1_pose.Q, 4, local_parameterization);
    problem.AddParameterBlock(kf2_pose.Q, 4, local_parameterization);
    problem.AddParameterBlock(kf1_pose.t, 3);
    problem.AddParameterBlock(kf2_pose.t, 3);
    problem.SetParameterBlockConstant(kf1_pose.Q);
    problem.SetParameterBlockConstant(kf1_pose.t);

    const Features &fts1 = kf1->fts_;
    std::vector<MapPointPose> mpts_pose;
    mpts_pose.reserve(fts1.size());

    for(int id = 0; id < fts1.size();id++)
    {
        Feature::Ptr ft1 = fts1[id];
        MapPoint::Ptr mpt = ft1->mpt;
        if(mpt == nullptr)
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        if(ft2 == nullptr || ft2->mpt == nullptr)
            continue;

        mpts_pose.push_back(MapPointPose(mpt, mpt->getPose()));

        ceres::CostFunction* cost_function1 = ssvo::ReprojectionError::Create(ft1->ft[0], ft1->ft[1]);
        problem.AddResidualBlock(cost_function1, NULL, kf1_pose.Q, kf1_pose.t, mpts_pose.back().t);

        ceres::CostFunction* cost_function2 = ssvo::ReprojectionError::Create(ft2->ft[0], ft2->ft[1]);
        problem.AddResidualBlock(cost_function2, NULL, kf2_pose.Q, kf2_pose.t, mpts_pose.back().t);
    }

    ceres::Solver::Options options;
    setOptions(options, true);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //! update pose
    //kf1_pose.update();
    kf2_pose.update();

    std::vector<MapPointPose>::iterator mpts_pose_iter = mpts_pose.begin();
    std::vector<MapPointPose>::iterator mpts_pose_end = mpts_pose.end();
    for(; mpts_pose_iter != mpts_pose_end ; ++mpts_pose_iter)
    {
        mpts_pose_iter->update();
    }

    sloveReport(summary, true);
}

}


}