#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

using namespace Eigen;

namespace ssvo {

class Camera {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    Camera(int width, int height, float fx, float fy, float cx, float cy,
           float k1 = 0.0, float k2 = 0.0, float p1 = 0.0, float p2 = 0.0);

    Camera(int width, int height, cv::Mat& K, cv::Mat& D);

    Vector3f lift(Vector2f& px) const;

    void liftPoints(std::vector<Vector2f>& pxs, std::vector<Vector3f>& fts) const;

    Vector2f project(Vector3f& P) const;

    inline const int width() { return width_; }

    inline const int height() { return height_; }

    inline const float fx() { return fx_; };

    inline const float fy() { return fy_; };

    inline const float cx() { return cx_; };

    inline const float cy() { return cy_; };

    inline const float d0() { return k1_; };

    inline const float d1() { return k2_; };

    inline const float d2() { return p1_; };

    inline const float d3() { return p2_; };

    inline cv::Mat cvK() { return cvK_; }

    inline const Matrix3f K() { return K_; }

    inline const Matrix3f Kinv() { return K_inv_; }


private:
    int width_;
    int height_;
    float fx_, fy_, cx_, cy_;
    float k1_, k2_, p1_, p2_;
    bool distortion_;

    cv::Mat cvK_;
    cv::Mat cvK_inv_;
    cv::Mat cvD_;

    Matrix3f K_;
    Matrix3f K_inv_;

};

typedef std::shared_ptr<Camera> CameraPtr;

}

#endif