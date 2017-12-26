#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"

namespace ssvo{

//! modified from SVO, https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/depth_filter.h#L35
/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int seed_counter;
    int id;                      //!< Seed ID, only used for visualization.
    KeyFrame::Ptr kf;            //!< Batch id is the id of the keyframe for which the seed was created.
    Feature::Ptr ftr;            //!< Feature in the keyframe for which the depth should be computed.
    double a;                    //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                    //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                   //!< Mean of normal distribution.
    double z_range;              //!< Max range of the possible depth.
    double sigma2;               //!< Variance of normal distribution.
    Matrix2d patch_cov;          //!< Patch covariance in reference image.
    Seed(KeyFrame::Ptr kf, Feature::Ptr ftr, double depth_mean, double depth_min);
    double computeTau(const SE3d& T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);
    void update(const double x, const double tau2);
};


class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void run();

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points);

    void insertNewFrame(Frame::Ptr frame, KeyFrame::Ptr keyframe, double mean_depth, double min_depth);

    static LocalMapper::Ptr create(const FastDetector::Ptr &fast_detector, double fps, bool report = false)
    { return LocalMapper::Ptr(new LocalMapper(fast_detector, fps, report));}

private:

    LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report = false);

    bool checkNewFrame();

    bool processNewKeyFrame();

    bool processNewFrame();

public:

    Map::Ptr map_;

private:

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    std::deque<std::pair<Frame::Ptr, KeyFrame::Ptr> > frames_buffer_;
    std::deque<std::pair<double, double> > depth_buffer_;
    std::deque<Seed, aligned_allocator<Seed> > seeds_;

    std::pair<Frame::Ptr, KeyFrame::Ptr> current_frame_;
    std::pair<double, double> current_depth_;

    const int delay_;
    const bool report_;

    std::mutex mutex_kfs_;
    std::condition_variable cond_process_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
