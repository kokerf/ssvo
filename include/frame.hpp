#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "seed.hpp"
#include "feature_detector.hpp"

namespace ssvo{

class KeyFrame;

class Frame : public noncopyable
{
public:

    typedef std::shared_ptr<Frame> Ptr;

    const ImgPyr images() const;

    const ImgPyr opticalImages() const;

    const cv::Mat getImage(int level) const;

    //! Transform (c)amera from (w)orld
    SE3d Tcw();

    //! Transform (w)orld from (c)amera
    SE3d Twc();

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

    bool isVisiable(const Vector3d &xyz_w, const int border = 0);

    //! Feature created by MapPoint
    int featureNumber();

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> features();

    void getFeatures(std::vector<Feature::Ptr> &fts);

    void getMapPoints(std::list<MapPoint::Ptr> &mpts);

    bool addFeature(const Feature::Ptr &ft);

    bool removeFeature(const Feature::Ptr &ft);

    bool removeMapPoint(const MapPoint::Ptr &mpt);

    Feature::Ptr getFeatureByMapPoint(const MapPoint::Ptr &mpt);

    //! Feature created by Seed
    int seedNumber();

    void getSeeds(std::vector<Feature::Ptr> &fts);

    bool addSeed(const Feature::Ptr &ft);

    bool removeSeed(const Seed::Ptr &seed);

    bool hasSeed(const Seed::Ptr &seed);

    bool getSceneDepth(double &depth_mean, double &depth_min);

    std::map<std::shared_ptr<KeyFrame>, int> getOverLapKeyFrames();

    inline void setRefKeyFrame(const std::shared_ptr<KeyFrame> &kf) {ref_keyframe_ = kf;}

    inline std::shared_ptr<KeyFrame> getRefKeyFrame() const {return ref_keyframe_;}

    inline static Ptr create(const cv::Mat& img, const double timestamp, Camera::Ptr cam)
    { return Ptr(new Frame(img, timestamp, cam)); }

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

    Frame(const ImgPyr& img_pyr, const uint64_t id, const double timestamp, const Camera::Ptr &cam);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;
    const double timestamp_;

    Camera::Ptr cam_;

    const int max_level_;
    static const cv::Size optical_win_size_;
    static float light_affine_a_;
    static float light_affine_b_;

    SE3d optimal_Tcw_;//! for optimization

protected:

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts_;

    std::unordered_map<Seed::Ptr, Feature::Ptr> seed_fts_;

    ImgPyr img_pyr_;

    SE3d Tcw_;
    SE3d Twc_;
    Vector3d Dw_;

    std::shared_ptr<KeyFrame> ref_keyframe_;

    std::mutex mutex_pose_;
    std::mutex mutex_feature_;
    std::mutex mutex_seed_;

private:
    ImgPyr optical_pyr_;
};

}

#endif