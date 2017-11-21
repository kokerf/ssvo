#ifndef _BUNDLE_ADJUSTMENT_HPP_
#define _BUNDLE_ADJUSTMENT_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"

namespace ceres {

using Sophus::SE3d;
using Sophus::Vector6d;

class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw,
                      double* T_plus_delta_raw) const {
        Eigen::Map<SE3d const> const T(T_raw);
        Eigen::Map<Vector6d const> const delta(delta_raw);
        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * SE3d::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        Eigen::Map<SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
        jacobian = T.internalJacobian().transpose();
        return true;
    }

    virtual int GlobalSize() const { return SE3d::num_parameters; }

    virtual int LocalSize() const { return SE3d::DoF; }
};

}


namespace ssvo {

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
            : observed_x_(observed_x), observed_y_(observed_y) {}

    template<typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        Sophus::SE3<T> pose = Eigen::Map<const Sophus::SE3<T> >(camera);
        Eigen::Matrix<T, 3, 1> p = Eigen::Map<const Eigen::Matrix<T, 3, 1> >(point);

        Eigen::Matrix<T, 3, 1> p1 = pose.rotationMatrix() * p + pose.translation();

        T predicted_x = (T)p1[0] / p1[2];
        T predicted_y = (T)p1[1] / p1[2];
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, Sophus::SE3d::num_parameters, 3>(
                new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x_;
    double observed_y_;
};

namespace BA
{

struct KeyFramePose{
    KeyFramePose(KeyFrame::Ptr keyframe, Sophus::SE3d T)
    {
        kf = keyframe;
        se3 = T;
    }

    inline void update()
    {
        kf->setPose(se3);
    }

    KeyFrame::Ptr kf;
    Sophus::SE3d se3;
};

struct MapPointPose{
    MapPointPose(MapPoint::Ptr mappoint, Vector3d translation)
    {
        mpt = mappoint;

        t[0] = translation[0];
        t[1] = translation[1];
        t[2] = translation[2];
    }

    inline void update()
    {
        mpt->setPose(t[0], t[1], t[2]);
    }

    MapPoint::Ptr mpt;
    double t[3];
};

void twoViewBA(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, Map::Ptr map);

void setOptions(ceres::Solver::Options& options, bool output = false);

void sloveReport(ceres::Solver::Summary& summary, bool output = false);

Vector2d reprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id);

}

}

#endif