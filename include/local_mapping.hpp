#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"
#include "alignment.hpp"

namespace ssvo{

//! modified from SVO, https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/depth_filter.h#L35
/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int seed_counter;
    int id;                      //!< Seed ID, only used for visualization.
    Feature::Ptr ft;             //!< Feature in the keyframe for which the depth should be computed.
    double a;                    //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                    //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                   //!< Mean of normal distribution.
    double z_range;              //!< Max range of the possible depth.
    double sigma2;               //!< Variance of normal distribution.
    Matrix2d patch_cov;          //!< Patch covariance in reference image.
    Seed(Feature::Ptr ft, double depth_mean, double depth_min);
    double computeTau(const SE3d& T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);
    void update(const double x, const double tau2);
};

typedef std::list<Seed, aligned_allocator<Seed> > Seeds;

class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void startThread();

    void stopThread();

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points);

    void insertNewFrame(Frame::Ptr frame, KeyFrame::Ptr keyframe, double mean_depth, double min_depth);

    static LocalMapper::Ptr create(const FastDetector::Ptr &fast_detector, double fps, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(fast_detector, fps, report, verbose));}

private:

    LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report, bool verbose);

    void run();

    void setStop();

    bool isRequiredStop();

    bool checkNewFrame();

    bool processNewKeyFrame();

    bool processNewFrame();

    bool findEpipolarMatch(const Seed &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame, const SE3d &T_cur_from_ref, double &depth);

    bool triangulate(const Matrix3d& R_cr,  const Vector3d& t_cr, const Vector3d& fn_r, const Vector3d& fn_c, double &d_ref);

public:

    Map::Ptr map_;

private:

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    std::deque<std::pair<Frame::Ptr, KeyFrame::Ptr> > frames_buffer_;
    std::deque<std::pair<double, double> > depth_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds>> > seeds_buffer_;

    std::pair<Frame::Ptr, KeyFrame::Ptr> current_frame_;
    std::pair<double, double> current_depth_;

    const int delay_;
    const bool report_;
    const bool verbose_;

    bool stop_require_;
    std::mutex mutex_kfs_;
    std::mutex mutex_stop_;
    std::condition_variable cond_process_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
