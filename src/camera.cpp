#include <assert.h>
#include "camera.hpp"

namespace ssvo {

Camera::Camera(int width, int height, float fx, float fy, float cx, float cy,
           float k1, float k2, float p1, float p2) :
            width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
            k1_(k1), k2_(k2), p1_(p1), p2_(p2), distortion_(fabs(k1_) > 0.0000001)
{
    cvK_ = (cv::Mat_<float>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    cvK_inv_ = cvK_.inv();
    K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
    K_inv_ = K_.inverse();
    cvD_ = (cv::Mat_<float>(1, 4) << k1_, k2_, p1_, p2_);
}

Camera::Camera(int width, int height, cv::Mat& K, cv::Mat& D):
        width_(width), height_(height)
{
    assert(K.cols == 3 && K.rows == 3);
    assert(D.cols == 1 || D.rows == 1);
    cvK_ = K.clone();
    cvK_inv_ = cvK_.inv();
    cvD_ = D.clone();

    fx_ = K.at<float>(0,0);
    fy_ = K.at<float>(1,1);
    cx_ = K.at<float>(0,2);
    cy_ = K.at<float>(1,2);

    const float* D_ptr = D.ptr<float>(0);
    k1_ = *D_ptr;
    k2_ = *(D_ptr+1);
    p1_ = *(D_ptr+2);
    p2_ = *(D_ptr+3);

    distortion_ = (fabs(k1_) > 0.0000001);
}

Vector3f Camera::lift(Vector2f &px) const
{
    float x = (px[0] - cx_) / fx_;
    float y = (px[0] - cy_) / fy_;
    if(distortion_)
    {
        float pt_u_arr[2] = {x, y};
        float pt_d_arr[2];
        cv::Mat pt_u(1, 1, CV_32FC2, pt_u_arr);
        cv::Mat pt_d(1, 1, CV_32FC2, pt_d_arr);
        cv::undistortPoints(pt_u, pt_d, cvK_, cvD_);
        x = pt_d_arr[0];
        y = pt_d_arr[1];
    }

    return Vector3f(x,y,1);
}

Vector2f Camera::project(Vector3f &P) const
{
    Vector2f px = P.head<2>() / P[2];
    if(distortion_)
    {
        const float x = px[0];
        const float y = px[1];
        const float x2 = x * x;
        const float y2 = y * y;
        const float r2 = x2 + y2;
        const float rdist = 1 + r2 * (k1_ + k2_ * r2);
        const float a1 = 2 * x * y;
        const float a2 = r2 + 2 * x * x;
        const float a3 = r2 + 2 * y * y;

        px[0] = x * rdist + p1_ * a1 + p2_ * a2;
        px[1] = y * rdist + p1_ * a3 + p2_ * a1;
    }

    px[0] = fx_ * px[0] + cx_;
    px[1] = fy_ * px[1] + cy_;
    return px;
}

}