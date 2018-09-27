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
class AbstractCamera : public noncopyable
{
public:

    enum Model {
        UNKNOW      = -2,
        ABSTRACT    = -1,
        PINHOLE     = 0,
        ATAN        = 1
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<AbstractCamera> Ptr;

    AbstractCamera();

    AbstractCamera(Model model = ABSTRACT);

    AbstractCamera(int width, int height, Model model = ABSTRACT);

    AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, Model model = ABSTRACT);

    virtual ~AbstractCamera() {};

    static Model checkCameraModel(std::string calib_file);

    inline const int fps() const { return fps_; }

    inline const int width() const { return width_; }

    inline const int height() const { return height_; }

    inline const double fx() const { return fx_; };

    inline const double fy() const { return fy_; };

    inline const double cx() const { return cx_; };

    inline const double cy() const { return cy_; };

    inline const cv::Mat K() const { return K_; };

    inline const cv::Mat D() const { return D_; };

    inline const cv::Mat T_BC() const { return T_BC_; };

    inline const Model model() const { return model_; }

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline bool isInFrame(const Vector2i &obs, int boundary=0) const
    {
        if(obs[0] >= boundary && obs[0] < width() - boundary
            && obs[1] >= boundary && obs[1] < height() - boundary)
            return true;
        return false;
    }

    inline bool isInFrame(const Vector2i &obs, int boundary, double scale_factor) const
    {
        if(obs[0] >= boundary && obs[0] < std::round(width() * scale_factor) - boundary
            && obs[1] >= boundary && obs[1] < std::round(height() * scale_factor) - boundary)
            return true;
        return false;
    }

protected:
    const Model model_;
    double fps_;
    int width_;
    int height_;
    double fx_, fy_, cx_, cy_;
    cv::Mat K_, D_;
    cv::Mat T_BC_;
    bool distortion_;
};

class PinholeCamera : public AbstractCamera
{

public:

    typedef std::shared_ptr<PinholeCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    //! all undistort points are in the normlized plane
    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline static PinholeCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static PinholeCamera::Ptr create(int width, int height, const cv::Mat& K, const cv::Mat& D)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, K, D));}

    inline static PinholeCamera::Ptr create(std::string calib_file)
    {return PinholeCamera::Ptr(new PinholeCamera(calib_file));}

private:

    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D);

    PinholeCamera(std::string calib_file);

private:

    double k1_, k2_, p1_, p2_;

};

class AtanCamera : public AbstractCamera
{

public:

    typedef std::shared_ptr<AtanCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline static AtanCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double s = 0.0)
    {return AtanCamera::Ptr(new AtanCamera(width, height, fx, fy, cx, cy, s));}

    inline static AtanCamera::Ptr create(int width, int height, const cv::Mat& K, const double s = 0.0)
    {return AtanCamera::Ptr(new AtanCamera(width, height, K, s));}

    inline static AtanCamera::Ptr create(std::string calib_file)
    {return AtanCamera::Ptr(new AtanCamera(calib_file));}

private:

    AtanCamera(int width, int height, double fx, double fy, double cx, double cy, double s = 0.0);

    AtanCamera(int width, int height, const cv::Mat& K, const double s = 0.0);

    AtanCamera(std::string calib_file);

private:

    double s_;
    double tans_;
};

}

#endif