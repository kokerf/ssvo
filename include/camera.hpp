#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

using namespace Eigen;

namespace ssvo {

class Camera {
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Camera> Ptr;

    Camera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    Camera(int width, int height, cv::Mat& K, cv::Mat& D);

    Vector3d lift(Vector2d& px) const;

    void liftPoints(std::vector<Vector2f>& pxs, std::vector<Vector3f>& fts) const;

    Vector2d project(Vector3d& P) const;

    inline const int width() { return width_; }

    inline const int height() { return height_; }

    inline const double fx() { return fx_; };

    inline const double fy() { return fy_; };

    inline const double cx() { return cx_; };

    inline const double cy() { return cy_; };

    inline const double d0() { return k1_; };

    inline const double d1() { return k2_; };

    inline const double d2() { return p1_; };

    inline const double d3() { return p2_; };

    inline cv::Mat cvK() { return cvK_; }

    inline cv::Mat cvD() { return cvD_; }

    inline const Matrix3d K() { return K_; }

    inline const Matrix3d Kinv() { return K_inv_; }

    inline static Camera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return Camera::Ptr(new Camera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static Camera::Ptr create(int width, int height, cv::Mat& K, cv::Mat& D)
    {return Camera::Ptr(new Camera(width, height, K, D));}


private:
    int width_;
    int height_;
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_;
    bool distortion_;

    cv::Mat cvK_;
    cv::Mat cvK_inv_;
    cv::Mat cvD_;

    Matrix3d K_;
    Matrix3d K_inv_;

};

}

#endif