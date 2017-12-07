#include <assert.h>
#include "camera.hpp"

namespace ssvo {

Camera::Camera(int width, int height, double fx, double fy, double cx, double cy,
           double k1, double k2, double p1, double p2) :
            width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
            k1_(k1), k2_(k2), p1_(p1), p2_(p2), distortion_(fabs(k1_) > 0.0000001)
{
    cvK_ = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cvK_inv_ = cvK_.inv();
    K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
    K_inv_ = K_.inverse();
    cvD_ = (cv::Mat_<double>(1, 4) << k1_, k2_, p1_, p2_);
}

Camera::Camera(int width, int height, const cv::Mat& K, const cv::Mat& D):
        width_(width), height_(height)
{
    assert(K.cols == 3 && K.rows == 3);
    assert(D.cols == 1 || D.rows == 1);
    if(K.type() == CV_64FC1)
        cvK_ = K.clone();
    else
        K.convertTo(cvK_, CV_64FC1);

    cvK_inv_ = cvK_.inv();

    if(D.type() == CV_64FC1)
        cvD_ = D.clone();
    else
        D.convertTo(cvD_, CV_64FC1);

    fx_ = K.at<double>(0,0);
    fy_ = K.at<double>(1,1);
    cx_ = K.at<double>(0,2);
    cy_ = K.at<double>(1,2);

    const double* D_ptr = D.ptr<double>(0);
    k1_ = *D_ptr;
    k2_ = *(D_ptr+1);
    p1_ = *(D_ptr+2);
    p2_ = *(D_ptr+3);

    distortion_ = (fabs(k1_) > 0.0000001);
}

Vector3d Camera::lift(const Vector2d &px) const
{
    Vector3d xyz(0, 0, 1);
    if(distortion_)
    {
        double p[2] = {px[0], px[1]};
        cv::Mat pt_d = cv::Mat(1, 1, CV_64FC2, p);
        cv::Mat pt_u = cv::Mat(1, 1, CV_64FC2, xyz.data());
        cv::undistortPoints(pt_d, pt_u, cvK_, cvD_);
    }
    else
    {
        xyz[0] = (px[0] - cx_) / fx_;
        xyz[1] = (px[0] - cy_) / fy_;
    }

    return xyz.normalized();
}

Vector2d Camera::project(const Vector3d &P) const
{
    Vector2d px = P.head<2>() / P[2];
    if(distortion_)
    {
        const double x = px[0];
        const double y = px[1];
        const double x2 = x * x;
        const double y2 = y * y;
        const double r2 = x2 + y2;
        const double rdist = 1 + r2 * (k1_ + k2_ * r2);
        const double a1 = 2 * x * y;
        const double a2 = r2 + 2 * x * x;
        const double a3 = r2 + 2 * y * y;

        px[0] = x * rdist + p1_ * a1 + p2_ * a2;
        px[1] = y * rdist + p1_ * a3 + p2_ * a1;
    }

    px[0] = fx_ * px[0] + cx_;
    px[1] = fy_ * px[1] + cy_;
    return px;
}

}