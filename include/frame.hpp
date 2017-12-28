#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "feature_detector.hpp"

namespace ssvo{

class KeyFrame;

class Frame : public noncopyable
{
public:

    typedef std::shared_ptr<Frame> Ptr;

    const ImgPyr image() const;

    const cv::Mat getImage(int level) const;

    //! Transform (c)amera from (w)orld
    SE3d Tcw();

    //! Transform (w)orld from (c)amera
    SE3d pose();

    //! Principal ray in world frame
    Vector3d ray();

    //! Set pose in world frame
    void setPose(const SE3d& pose);

    //! Set pose in world frame
    void setPose(const Matrix3d& R, const Vector3d& t);

    //! Set Extrinsic Matrix
    void setTcw(const SE3d& Tcw);

    bool isVisiable(const Vector3d &xyz_w);

    Features features();

    std::vector<Feature::Ptr> getFeatures();

    void addFeature(const Feature::Ptr ft);

    bool getSceneDepth(double &depth_mean, double &depth_min);

    std::map<std::shared_ptr<KeyFrame>, int> getOverLapKeyFrames();

    inline int N() const {return fts_.size();}

    inline void setRefKeyFrame(const std::shared_ptr<KeyFrame> &kf) {ref_keyframe_ = kf;}

    inline std::shared_ptr<KeyFrame> getRefKeyFrame() const {return ref_keyframe_;}

    inline static Ptr create(const cv::Mat& img, const double timestamp, Camera::Ptr cam)
    { return Ptr(new Frame(img, timestamp, cam)); }

    inline static Ptr create(const ImgPyr& img_pyr, const double timestamp, Camera::Ptr cam)
    { return Ptr(new Frame(img_pyr, timestamp, cam)); }

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

    Frame(const cv::Mat& img, const double timestamp, const Camera::Ptr &cam);

    Frame(const ImgPyr& img_pyr, const double timestamp, const Camera::Ptr &cam);

    Frame(const ImgPyr& img_pyr, const uint64_t id, const double timestamp, const Camera::Ptr &cam);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;
    const double timestamp_;

    Camera::Ptr cam_;

    const int nlevels_;

    SE3d optimal_Tcw_;//! for optimization

protected:

    Features fts_;

    ImgPyr img_pyr_;

    SE3d Tcw_;
    SE3d Twc_;
    Vector3d Dw_;

    std::shared_ptr<KeyFrame> ref_keyframe_;

    std::mutex mutex_pose_;
    std::mutex mutex_feature_;
};

}

#endif