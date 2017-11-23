#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "global.hpp"

using namespace Eigen;

namespace ssvo {

// once created, never changed
class Camera : public noncopyable
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Camera> Ptr;

    Vector3d lift(const Vector2d& px) const;

    void liftPoints(std::vector<Vector2f>& pxs, std::vector<Vector3f>& fts) const;

    Vector2d project(const Vector3d& P) const;

    inline const int width() const { return width_; }

    inline const int height() const { return height_; }

    inline const double fx() const { return fx_; };

    inline const double fy() const { return fy_; };

    inline const double cx() const { return cx_; };

    inline const double cy() const { return cy_; };

    inline const double d0() const { return k1_; };

    inline const double d1() const { return k2_; };

    inline const double d2() const { return p1_; };

    inline const double d3() const { return p2_; };

    inline cv::Mat cvK() const { return cvK_; }

    inline cv::Mat cvD() const { return cvD_; }

    inline const Matrix3d K() const { return K_; }

    inline const Matrix3d Kinv() const { return K_inv_; }

    inline bool observable(const Vector2i & obs, int boundary=0) const
    {
        if(obs[0]>=boundary && obs[0]<width()-boundary
            && obs[1]>=boundary && obs[1]<height()-boundary)
            return true;
        return false;
    }

    inline static Camera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return Camera::Ptr(new Camera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static Camera::Ptr create(int width, int height, const cv::Mat& K, const cv::Mat& D)
    {return Camera::Ptr(new Camera(width, height, K, D));}

private:
    Camera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    Camera(int width, int height, const cv::Mat& K, const cv::Mat& D);

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