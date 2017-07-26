#ifndef BUNDLE_ADJUSTMENT
#define BUNDLE_ADJUSTMENT

#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
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

class GlobalBA {
public:
    GlobalBA();

    ~GlobalBA();

    void addFrames(std::vector<FramePtr> &frames);

    void slove();

private:

};

}

#endif