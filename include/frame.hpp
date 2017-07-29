#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "feature_detector.hpp"

namespace ssvo{

typedef std::vector<Feature::Ptr> Features;
typedef std::vector<MapPoint::Ptr> MapPoints;

int createPyramid(const cv::Mat& img, ImgPyr& img_pyr, const uint16_t nlevels = 4, const cv::Size min_size = cv::Size(40, 40));

class Frame
{

public:

    typedef std::shared_ptr<Frame> Ptr;

    Frame() = default;

    Frame(const cv::Mat& img, const double timestamp, Camera::Ptr cam);

    Frame(const ImgPyr& img_pyr, const double timestamp, Camera::Ptr cam);

    Frame &operator=(const Frame&) = delete; //! copy denied

    void addFeature(const Feature::Ptr ft);

    inline void setPose(const MatrixXd T)
    {
        Matrix3d R = T.block(0,0,3,3);
        q_ = Quaterniond(R);
        t_ = T.block(0,3,3,1);
    }

    inline void setRotation(const double w, const double x, const double y, const double z)
    {
        q_.w() = w;
        q_.x() = x;
        q_.y() = y;
        q_.z() = z;
    }

    inline void setTranslation(const double x, const double y, const double z){
        t_[0] = x;
        t_[1] = y;
        t_[2] = z;
    }

    inline void setRotation(const Quaterniond q) {q_= q;}

    inline void setTranslation(const Vector3d t) {t_ = t;}

    inline Quaterniond getRotation() {return q_;}

    inline Vector3d getTranslation() {return t_;}

    inline long unsigned int id() {return id_;}

    inline double timeStamp() {return timestamp_;}

    inline static Frame::Ptr create(const cv::Mat& img, const double timestamp, Camera::Ptr cam) {return Frame::Ptr(new Frame(img, timestamp, cam));}

    inline static Frame::Ptr create(const ImgPyr& img_pyr, const double timestamp, Camera::Ptr cam) {return Frame::Ptr(new Frame(img_pyr, timestamp, cam));}

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;

    Camera::Ptr cam_;
    ImgPyr img_pyr_;
    Features fts_;
    MapPoints mpts_;

protected:

    uint64_t id_;
    double timestamp_;

    Quaterniond q_;
    Vector3d t_;

};

}

#endif