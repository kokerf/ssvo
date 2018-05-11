#include "camera.hpp"

namespace ssvo {

//! =========================
//! AbstractCamera
//! =========================
AbstractCamera::AbstractCamera(int width, int height, Type type) :
    width_(width), height_(height), type_(type) {}

AbstractCamera::AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, Type type) :
    width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy), type_(type) {}

Vector3d AbstractCamera::lift(const Vector2d& px) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

Vector3d AbstractCamera::lift(double x, double y) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

Vector2d AbstractCamera::project(const Vector3d& xyz) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

Vector2d AbstractCamera::project(double x, double y) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

void AbstractCamera::undistortPoints(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//! =========================
//! PinholeCamera
//! =========================
PinholeCamera::PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1, double k2, double p1, double p2) :
            AbstractCamera(width, height, fx, fy, cx, cy, PINHOLE),
            k1_(k1), k2_(k2), p1_(p1), p2_(p2)
{
    distortion_ = (fabs(k1_) > 0.0000001);
}

PinholeCamera::PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D):
        AbstractCamera(width, height, PINHOLE)
{
    assert(K.cols == 3 && K.rows == 3);
    assert(D.cols == 1 || D.rows == 1);
    if(K.type() == CV_64FC1)
        cvK_ = K.clone();
    else
        K.convertTo(cvK_, CV_64FC1);

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

//! return the px lift to normalized plane
Vector3d PinholeCamera::lift(const Vector2d &px) const
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
        xyz[1] = (px[1] - cy_) / fy_;
    }

    return xyz;
}

Vector3d PinholeCamera::lift(double x, double y) const
{
    Vector3d xyz(0, 0, 1);
    if(distortion_)
    {
        double p[2] = {x, y};
        cv::Mat pt_d = cv::Mat(1, 1, CV_64FC2, p);
        cv::Mat pt_u = cv::Mat(1, 1, CV_64FC2, xyz.data());
        cv::undistortPoints(pt_d, pt_u, cvK_, cvD_);
    }
    else
    {
        xyz[0] = (x - cx_) / fx_;
        xyz[1] = (y - cy_) / fy_;
    }

    return xyz;
}

Vector2d PinholeCamera::project(const Vector3d &xyz) const
{
    Vector2d px = xyz.head<2>() / xyz[2];
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

Vector2d PinholeCamera::project(double x, double y) const
{
    Vector2d px(x, y);
    if(distortion_)
    {
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

void PinholeCamera::undistortPoints(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst) const
{
    cv::undistortPoints(src, dst, cvK_, cvD_);
}

}
