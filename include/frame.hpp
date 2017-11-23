#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "feature_detector.hpp"

namespace ssvo{

int createPyramid(const cv::Mat& img, ImgPyr& img_pyr, const uint16_t nlevels = 4, const cv::Size min_size = cv::Size(40, 40));

class Frame: public noncopyable
{
public:

    typedef std::shared_ptr<Frame> Ptr;

    inline Features getFeatures() { return fts_; }

    inline void addFeature(const Feature::Ptr ft) { fts_.push_back(ft); };

    inline void setPose(const Sophus::SE3d& T) { Tw_ = T; }

    inline void setPose(const Matrix3d& R, const Vector3d& t) { Tw_ = Sophus::SE3d(R, t); }

    inline Sophus::SE3d pose() { return Tw_; }

    inline static Frame::Ptr create(const cv::Mat& img, const double timestamp, Camera::Ptr cam)
    { return std::make_shared<Frame>(Frame(img, timestamp, cam)); }

    inline static Frame::Ptr create(const ImgPyr& img_pyr, const double timestamp, Camera::Ptr cam)
    { return std::make_shared<Frame>(Frame(img_pyr, timestamp, cam)); }

protected:
    Frame(const cv::Mat& img, const double timestamp, const Camera::Ptr cam);

    Frame(const ImgPyr& img_pyr, const double timestamp, const Camera::Ptr cam);

    Frame(const ImgPyr& img_pyr, const uint64_t id, const double timestamp, const Camera::Ptr cam);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;
    const double timestamp_;

    Camera::Ptr cam_;

    ImgPyr img_pyr_;

protected:

    Features fts_;

    Sophus::SE3d Tw_;

};

}

#endif