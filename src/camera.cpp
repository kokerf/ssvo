#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include "camera.hpp"

namespace ssvo {

//! =========================
//! AbstractCamera
//! =========================
AbstractCamera::AbstractCamera() :
        model_(ABSTRACT) {}

AbstractCamera::AbstractCamera(Model model) :
        model_(model) {}

AbstractCamera::AbstractCamera(int width, int height, cv::Mat Tbc, Model type) :
        model_(type), width_(width), height_(height)
{
	cv:cv2eigen(Tbc, T_BC_);
}

AbstractCamera::AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, cv::Mat Tbc, Model model) :
        model_(model), width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
{
    K_.setIdentity();
    K_(0,0) = fx_;
    K_(0,2) = cx_;
    K_(1,1) = fy_;
    K_(1,2) = cy_;

	cv:cv2eigen(Tbc, T_BC_);
}

AbstractCamera::Model AbstractCamera::checkCameraModel(std::string calib_file)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;

    //! camera parameters
    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    fs.release();

    if(str_camera_model == "pinhole")
        return PINHOLE;
    else if(str_camera_model == "atan")
        return ATAN;
    else
        return UNKNOW;
}

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

void AbstractCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    LOG(FATAL) << "Please instantiation!!!";
}

//! =========================
//! PinholeCamera
//! =========================
PinholeCamera::PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1, double k2, double p1, double p2, cv::Mat Tbc) :
            AbstractCamera(width, height, fx, fy, cx, cy, Tbc, PINHOLE),
            k1_(k1), k2_(k2), p1_(p1), p2_(p2)
{
    distortion_ = (fabs(k1_) > 0.0000001);
	D_.resize(4);
    D_(0) = k1;
    D_(1) = k2;
    D_(2) = p1;
    D_(3) = p2;
}

PinholeCamera::PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D, cv::Mat Tbc):
        AbstractCamera(width, height, Tbc, PINHOLE)
{
    assert(K.cols == 3 && K.rows == 3);
    assert(D.cols == 1 || D.rows == 1);

	cvK_ = K.clone();
	cvD_ = D.clone();

	cv::cv2eigen(K, K_);

	cv::cv2eigen(D, D_);

    fx_ = K.at<double>(0,0);
    fy_ = K.at<double>(1,1);
    cx_ = K.at<double>(0,2);
    cy_ = K.at<double>(1,2);

    k1_ = D.at<double>(0);
    k2_ = D.at<double>(1);
    p1_ = D.at<double>(2);
    p2_ = D.at<double>(3);

    distortion_ = (fabs(k1_) > 0.0000001);
}

PinholeCamera::PinholeCamera(std::string calib_file) :
    AbstractCamera(PINHOLE)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;

    //! camera parameters
    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    LOG_ASSERT(str_camera_model == "pinhole") << "Wrong camera modle: " << str_camera_model;

    fps_ = (double)fs["Camera.fps"];

    cv::FileNode resolution = fs["Camera.resolution"];
    LOG_ASSERT(resolution.size() == 2) << "Failed to load Camera.resolution with error size: " << resolution.size();
    width_ = resolution[0];
    height_ = resolution[1];

    cv::FileNode intrinsics = fs["Camera.intrinsics"];
    LOG_ASSERT(intrinsics.size() == 4) << "Failed to load Camera.intrinsics with error size: " << intrinsics.size();
    fx_ = intrinsics[0];
    fy_ = intrinsics[1];
    cx_ = intrinsics[2];
    cy_ = intrinsics[3];
    cvK_ = cv::Mat::eye(3,3,CV_64FC1);
    cvK_.at<double>(0,0) = fx_;
    cvK_.at<double>(0,2) = cx_;
    cvK_.at<double>(1,1) = fy_;
    cvK_.at<double>(1,2) = cy_;

    cv::cv2eigen(cvK_, K_);

    cv::FileNode distortion_coefficients = fs["Camera.distortion_coefficients"];
    LOG_ASSERT(distortion_coefficients.size() == 4) << "Failed to load Camera.distortion_coefficients with error size: " << distortion_coefficients.size();
    k1_ = distortion_coefficients[0];
    k2_ = distortion_coefficients[1];
    p1_ = distortion_coefficients[2];
    p2_ = distortion_coefficients[3];

	cvD_ = cv::Mat::zeros(1, 4, CV_64FC1);
	cvD_.at<double>(0) = k1_;
	cvD_.at<double>(1) = k2_;
	cvD_.at<double>(2) = p1_;
	cvD_.at<double>(3) = p2_;
    cv::cv2eigen(cvD_, D_);

    distortion_ = (fabs(k1_) > 0.0000001);

	cv::Mat T;
	fs["Camera.T_BC"] >> T;
    LOG_ASSERT(T.size() == cv::Size(4, 4)) << "Failed to load Camera.T_BC with error size: " << T.size();
	cv:cv2eigen(T, T_BC_);

    fs.release();
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

void PinholeCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    cv::undistortPoints(pts_dist, pts_udist, cvK_, cvD_);
}

//! =========================
//! AtanCamera
//! =========================
AtanCamera::AtanCamera(int width, int height, double fx, double fy, double cx, double cy, double s, cv::Mat Tbc) :
        AbstractCamera(width, height, width*fx, height*fy, width*cx-0.5, height*cy-0.5, Tbc, ATAN), s_(s)
{
	D_.resize(1);
    D_(0) = s_;

    if(fabs(s_) > 0.0000001)
    {
        tans_ = 2.0 * tan(s_ / 2.0);
        distortion_ = true;
    }
    else
    {
        tans_ = 0.0;
        distortion_ = false;
    }
}

AtanCamera::AtanCamera(int width, int height, const cv::Mat& K, const double s, cv::Mat Tbc):
        AbstractCamera(width, height, Tbc, ATAN), s_(s)
{
    assert(K.cols == 3 && K.rows == 3);

    fx_ = K.at<double>(0,0);
    fy_ = K.at<double>(1,1);
    cx_ = K.at<double>(0,2);
    cy_ = K.at<double>(1,2);

	D_.resize(1);
    D_(0) = s_;

    if(fabs(s_) > 0.0000001)
    {
        tans_ = 2.0 * tan(s_ / 2.0);
        distortion_ = true;
    }
    else
    {
        tans_ = 0.0;
        distortion_ = false;
    }
}

AtanCamera::AtanCamera(std::string calib_file) :
        AbstractCamera(ATAN)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    LOG_ASSERT(fs.isOpened()) << "Failed to open calibration file at: " << calib_file;

    //! camera parameters
    std::string str_camera_model;
    if (!fs["Camera.model"].empty())
        fs["Camera.model"] >> str_camera_model;

    LOG_ASSERT(str_camera_model == "atan") << "Wrong camera modle: " << str_camera_model;

    fps_ = (double)fs["Camera.fps"];

    cv::FileNode resolution = fs["Camera.resolution"];
    LOG_ASSERT(resolution.size() == 2) << "Failed to load Camera.resolution with error size: " << resolution.size();
    width_ = resolution[0];
    height_ = resolution[1];

    cv::FileNode intrinsics = fs["Camera.intrinsics"];
    LOG_ASSERT(intrinsics.size() == 4) << "Failed to load Camera.intrinsics with error size: " << intrinsics.size();
    fx_ = intrinsics[0];
    fy_ = intrinsics[1];
    cx_ = intrinsics[2];
    cy_ = intrinsics[3];

    cv::FileNode distortion_coefficients = fs["Camera.distortion_coefficients"];
    LOG_ASSERT(distortion_coefficients.size() == 1) << "Failed to load Camera.distortion_coefficients with error size: " << distortion_coefficients.size();
    double s = distortion_coefficients[0];
    fx_ *= width_;
    fy_ *= height_;
    cx_ = cx_ * width_ - 0.5;
    cy_ = cy_ * height_ - 0.5;

    D_.resize(1);
    D_(0) = s;

    K_.setIdentity();
    K_(0,0) = fx_;
    K_(1,1) = fy_;
    K_(0,2) = cx_;
    K_(1,2) = cy_;

	cv::Mat T;
	fs["Camera.T_BC"] >> T;
	LOG_ASSERT(T.size() == cv::Size(4, 4)) << "Failed to load Camera.T_BC with error size: " << T.size();
	cv:cv2eigen(T, T_BC_);

    fs.release();
}

//! return the px lift to normalized plane
Vector3d AtanCamera::lift(const Vector2d &px) const
{
    Vector2d px_d((px[0]-cx_)/fx_, (px[1]-cy_)/fy_);
    if(distortion_)
    {
        const double r_d = px_d.norm();

        if(r_d > 0.01)
        {
            const double factor_d = tan(r_d * s_) / (r_d * tans_);
            px_d.array() *= factor_d;
        }
    }

    return  Vector3d(px_d[0], px_d[1], 1);
}

Vector3d AtanCamera::lift(double x, double y) const
{
    Vector2d px_d((x-cx_)/fx_, (y-cy_)/fy_);
    if(distortion_)
    {
        const double r_d = px_d.norm();

        if(r_d > 0.01)
        {
            const double factor_d = tan(r_d * s_) / (r_d * tans_);
            px_d.array() *= factor_d;
        }
    }

    return  Vector3d(px_d[0], px_d[1], 1);
}

Vector2d AtanCamera::project(const Vector3d &xyz) const
{
    Vector2d px = xyz.head<2>() / xyz[2];
    if(distortion_)
    {
        const double ru = px.norm();

        if(ru > 0.001)
        {
            const double factor_u = atan(tans_ * ru) / (ru * s_);

            px.array() *= factor_u;
        }
    }

    px[0] = fx_ * px[0] + cx_;
    px[1] = fy_ * px[1] + cy_;

    return px;
}

Vector2d AtanCamera::project(double x, double y) const
{
    Vector2d px(x, y);
    if(distortion_)
    {
        const double ru = px.norm();

        if(ru > 0.001)
        {
            const double factor_u = atan(tans_ * ru) / (ru * s_);

            px.array() *= factor_u;
        }
    }

    px[0] = fx_ * px[0] + cx_;
    px[1] = fy_ * px[1] + cy_;

    return px;
}

void AtanCamera::undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const
{
    pts_udist.resize(pts_dist.size());

    if(!distortion_)
    {
        pts_udist = pts_dist;
        return;
    }

    const size_t N = pts_dist.size();
    for(size_t i = 0; i < N; i++)
    {
        Vector3d fn = lift(pts_dist[i].x, pts_dist[i].y);
        pts_udist[i].x = fn[0];//static_cast<float>(fx_ * fn[0] + cx_);
        pts_udist[i].y = fn[1];//static_cast<float>(fy_ * fn[1] + cy_);
    }
}


}
