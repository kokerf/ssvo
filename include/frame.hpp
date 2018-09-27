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

    virtual ~Frame() {};

    typedef std::shared_ptr<Frame> Ptr;

    //! ----- Image ---------------------------------------------
    const ImgPyr images() const;

    const cv::Mat getImage(int level) const;

    const ImgPyr opticalImages();

    //! ----- Pose ----------------------------------------------
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

    //! ---- Feature & MapPoint & Seed ---------------------------
    size_t getMapPointMatchSize();

    size_t getSeedMatchSize();

    std::vector<size_t> getMapPointMatchIndices();

    std::vector<size_t> getSeedMatchIndices();

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> getMapPointFeaturesMatched();

    std::unordered_map<Seed::Ptr, Feature::Ptr> getSeedFeaturesMatched();

    bool removeMapPointMatchByIndex(const size_t &idx);

    bool removeSeedMatchByIndex(const size_t &idx);

    std::vector<Feature::Ptr> getFeatures();

    std::vector<MapPoint::Ptr> getMapPoints();

    std::vector<Seed::Ptr> getSeeds();

    bool addMapPointFeatureMatch(const MapPoint::Ptr &mpt, const Feature::Ptr &ft);

    bool addSeedFeatureMatch(const Seed::Ptr &seed, const Feature::Ptr &ft);

    //! ----- Scene ----------------------------------------------
    bool isVisiable(const Vector3d &xyz_w, const int border = 0);

    bool getSceneDepth(double &depth_median, double &depth_min);

    //! ----- KeyFrame -------------------------------------------
    std::map<std::shared_ptr<KeyFrame>, int> getOverLapKeyFrames();

    inline void setRefKeyFrame(const std::shared_ptr<KeyFrame> &kf) {ref_keyframe_ = kf;}

    inline std::shared_ptr<KeyFrame> getRefKeyFrame() const {return ref_keyframe_;}

    //! ----- Static Function ------------------------------------
    static void initScaleParameters(const FastDetector::Ptr &fast);

    inline static Ptr create(const cv::Mat& img, const double timestamp, AbstractCamera::Ptr cam)
    { return Ptr(new Frame(img, timestamp, cam)); }

protected:

    Frame(const cv::Mat& img, const double timestamp, const AbstractCamera::Ptr &cam);

    //! only for keyframe init
    Frame(const ImgPyr& img_pyr, const uint64_t id, const double timestamp, const AbstractCamera::Ptr &cam);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;
    const double timestamp_;

    AbstractCamera::Ptr cam_;

    static const cv::Size optical_win_size_;
    static float light_affine_a_;
    static float light_affine_b_;

    static bool isInit_;
    static int nlevels_;
    static double scale_factor_;
    static double log_scale_factor_;
    static std::vector<double> scale_factors_;
    static std::vector<double> inv_scale_factors_;
    static std::vector<double> level_sigma2_;
    static std::vector<double> inv_level_sigma2_;

    SE3d optimal_Tcw_;//! for optimization

protected:

    ImgPyr img_pyr_;

    std::vector<Feature::Ptr> fts_;
    std::vector<MapPoint::Ptr> mpts_;
    std::vector<Seed::Ptr> seeds_;

    std::unordered_set<size_t> mpts_matched_;
    std::unordered_set<size_t> seeds_matched_;

    SE3d Tcw_;
    SE3d Twc_;
    Vector3d Dw_;

    std::shared_ptr<KeyFrame> ref_keyframe_;

    std::mutex mutex_pose_;
    std::mutex mutex_feature_;

private:

    ImgPyr optical_pyr_;
};

}

#endif