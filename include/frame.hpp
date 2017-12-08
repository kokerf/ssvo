#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "feature_detector.hpp"

namespace ssvo{

class Frame//: private noncopyable
{
public:

    typedef std::shared_ptr<Frame> Ptr;

    const cv::Mat getImage(int level) const;

    const ImgPyr image() const {return img_pyr_;}

    inline Sophus::SE3d pose() const { return Tcw_; }

    inline Sophus::SE3d pose_inverse() const { return Twc_; }

    inline void setPose(const Sophus::SE3d& T) { Tcw_ = T; Twc_ = Tcw_.inverse();}

    inline void setPose(const Matrix3d& R, const Vector3d& t) { Tcw_ = Sophus::SE3d(R, t); Twc_ = Tcw_.inverse();}

    inline Features features() const {return fts_; }

    inline std::vector<Feature::Ptr> getFeatures() const {return std::vector<Feature::Ptr>(fts_.begin(), fts_.end()); }

    inline void addFeature(const Feature::Ptr ft) { fts_.push_back(ft); };

    inline static Ptr create(const cv::Mat& img, const double timestamp, Camera::Ptr cam)
    { return std::make_shared<Frame>(Frame(img, timestamp, cam)); }

    inline static Ptr create(const ImgPyr& img_pyr, const double timestamp, Camera::Ptr cam)
    { return std::make_shared<Frame>(Frame(img_pyr, timestamp, cam)); }

    inline static void jacobian_xyz2uv(
        const Vector3d& xyz_in_f,
        Matrix<double,2,6,RowMajor>& J)
    {
        const double x = xyz_in_f[0];
        const double y = xyz_in_f[1];
        const double z_inv = 1./xyz_in_f[2];
        const double z_inv_2 = z_inv*z_inv;

        J(0,0) = -z_inv;              // -1/z
        J(0,1) = 0.0;                 // 0
        J(0,2) = x*z_inv_2;           // x/z^2
        J(0,3) = y*J(0,2);            // x*y/z^2
        J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
        J(0,5) = y*z_inv;             // y/z

        J(1,0) = 0.0;                 // 0
        J(1,1) = -z_inv;              // -1/z
        J(1,2) = y*z_inv_2;           // y/z^2
        J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
        J(1,4) = -J(0,3);             // -x*y/z^2
        J(1,5) = -x*z_inv;            // x/z
    }

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

    Sophus::SE3d optimal_Tw_;//! for optimization

protected:

    Features fts_;

    ImgPyr img_pyr_;

    Sophus::SE3d Tcw_;
    Sophus::SE3d Twc_;
};

}

#endif