#ifndef BUNDLE_ADJUSTMENT
#define BUNDLE_ADJUSTMENT

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"

namespace ssvo {

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
            : observed_x_(observed_x), observed_y_(observed_y) {}

    template<typename T>
    bool operator()(const T *const Q_cam, const T *const t_cam, const T *const point, T *residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(Q_cam, point, p);
        p[0] += t_cam[0];
        p[1] += t_cam[1];
        p[2] += t_cam[2];
        T predicted_x = p[0] / p[2];
        T predicted_y = p[1] / p[2];
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
                new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x_;
    double observed_y_;
};

namespace BA
{

struct KeyFramePose{
    KeyFramePose(KeyFrame::Ptr keyframe, Quaterniond quaterniond, Vector3d translation)
    {
        kf = keyframe;

        Q[0] = quaterniond.w();
        Q[1] = quaterniond.x();
        Q[2] = quaterniond.y();
        Q[3] = quaterniond.z();

        t[0] = translation[0];
        t[1] = translation[1];
        t[2] = translation[2];
    }

    inline void update()
    {
        kf->setRotation(Q[0], Q[1], Q[2], Q[3]);
        kf->setTranslation(t[0], t[1], t[2]);
    }

    KeyFrame::Ptr kf;
    double Q[4];
    double t[3];
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

}

}

#endif