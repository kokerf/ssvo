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

    enum Type {
        ABSTRACT    = -1,
        PINHOLE     = 0,
        ATAN        = 1
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<AbstractCamera> Ptr;

    AbstractCamera() {}

    AbstractCamera(int width, int height, Type type = ABSTRACT);

    AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, Type type = ABSTRACT);

    virtual ~AbstractCamera() {};

    inline const int width() const { return width_; }

    inline const int height() const { return height_; }

    inline const double fx() const { return fx_; };

    inline const double fy() const { return fy_; };

    inline const double cx() const { return cx_; };

    inline const double cy() const { return cy_; };

    inline const Type type() const { return type_; }

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst) const;

    inline bool isInFrame(const Vector2i &obs, int boundary=0) const
    {
        if(obs[0] >= boundary && obs[0] < width() - boundary
            && obs[1] >= boundary && obs[1] < height() - boundary)
            return true;
        return false;
    }

    inline bool isInFrame(const Vector2i &obs, int boundary, int level) const
    {
        if(obs[0] >= boundary && obs[0] < (width() >> level) - boundary
            && obs[1] >= boundary && obs[1] < (height() >> level) - boundary)
            return true;
        return false;
    }

protected:
    int width_;
    int height_;
    double fx_, fy_, cx_, cy_;
    bool distortion_;
    Type type_;
};

class PinholeCamera : public AbstractCamera
{

public:

    typedef std::shared_ptr<PinholeCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst) const;

    inline static PinholeCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static PinholeCamera::Ptr create(int width, int height, const cv::Mat& K, const cv::Mat& D)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, K, D));}

private:

    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D);

private:

    double k1_, k2_, p1_, p2_;
    cv::Mat cvK_, cvD_;

};

}

#endif