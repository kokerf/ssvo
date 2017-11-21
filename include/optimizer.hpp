#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"

namespace ssvo {

// https://github.com/strasdat/Sophus/blob/v1.0.0/test/ceres/local_parameterization_se3.hpp
class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw,
                      double* T_plus_delta_raw) const {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Sophus::SE3d::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        Eigen::Map<Sophus::SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
        jacobian = T.internalJacobian().transpose();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

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


class Optimizer {
public:
    Optimizer();

    void twoViewBundleAdjustment(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, Map::Ptr map);

    void report(bool with_residual = false);

    static Vector2d reprojectionError(const ceres::Problem &problem, ceres::ResidualBlockId id);

private:
    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;
};

}

#endif