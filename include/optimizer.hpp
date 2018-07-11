#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "factor.hpp"
#include "preintegration.hpp"
#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"

namespace ssvo {

class Optimizer: public noncopyable
{
public:

    static void globleBundleAdjustment(const Map::Ptr &map, int max_iters, bool report=false, bool verbose=false);

    static void motionOnlyBundleAdjustment(const Frame::Ptr &frame, bool use_seeds, bool reject=false, bool report=false, bool verbose=false);

    static void localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int max_iters, int size=10, int min_shared_fts=50, bool report=false, bool verbose=false);

//    static void localBundleAdjustmentWithInvDepth(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size=10, bool report=false, bool verbose=false);

    static void refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report=false, bool verbose=false);

	//! for vio
	static bool sloveInitialGyroBias(const std::vector<Frame::Ptr> &frames, Vector3d &dbias_gyro, bool report = false, bool verbose = false);

    static bool sloveScaleAndGravity(const std::vector<Frame::Ptr> &frames, Vector4d &scale_and_gravity, double threshold = 1e3, bool verbose = false);

	static bool sloveInitialAccBiasAndRefine(const std::vector<Frame::Ptr> &frames, Vector4d &scale_and_gravity, Vector3d &dbias_acc, double threshold = 1e5, bool verbose = false);

	static bool initIMU(const std::vector<Frame::Ptr> &frames, VectorXd &result, bool report = false, bool verbose = false);


    template<int nRes>
    static inline Eigen::Matrix<double, nRes, 1> evaluateResidual(const ceres::Problem& problem, ceres::ResidualBlockId id)
    {
        auto cost = problem.GetCostFunctionForResidualBlock(id);
        std::vector<double*> parameterBlocks;
        problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
        Eigen::Matrix<double, nRes, 1> residual;
        cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
        return residual;
    }

    template<int nRes>
    static inline void reportInfo(const ceres::Problem &problem, const ceres::Solver::Summary summary, bool report = false, bool verbose = false)
    {
        if (!report) return;

        if (!verbose)
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
                LOG(INFO) << "BlockId: " << std::setw(5) << i << " residual(RMSE): " << evaluateResidual<nRes>(problem, ids[i]).norm();
            }
        }
    }
};

}//! namespace ssvo

#endif