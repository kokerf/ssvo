#ifndef _SSVO_DEPTH_FILTER_HPP_
#define _SSVO_DEPTH_FILTER_HPP_

#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"

namespace ssvo
{

//! modified from SVO, https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/depth_filter.h#L35
/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Seed> Ptr;

    const std::shared_ptr<KeyFrame> kf;     //!< Reference KeyFrame, where the seed created.
    const Vector3d fn_ref;                  //!< Pixel in the keyframe's normalized plane where the depth should be computed.
    const Vector2d px_ref;                  //!< Pixel matched in current frame
    const int level_ref;                    //!< Corner detected level in refrence frame
    Vector2d px_cur;                        //!< Pixel matched in current frame
    int level_cur;                          //!< Corner detected level in current frame
    double a;                               //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                               //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                              //!< Mean of normal distribution.
    double z_range;                         //!< Max range of the possible depth.
    double sigma2;                          //!< Variance of normal distribution.
    Matrix2d patch_cov;                     //!< Patch covariance in reference image.

    std::list<std::pair<double, double> > history;

    double computeTau(const SE3d &T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);
    double computeVar(const SE3d &T_cur_ref, const double z, const double delta);
    void update(const double x, const double tau2);
    inline static Ptr create(const std::shared_ptr<KeyFrame> &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min)
    {return std::make_shared<Seed>(Seed(kf, px, fn, level, depth_mean, depth_min));}

private:
    Seed(const std::shared_ptr<KeyFrame> &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min);
};

typedef std::list<Seed::Ptr> Seeds;


class DepthFilter : public noncopyable
{
public:

    typedef std::shared_ptr<DepthFilter> Ptr;

    typedef std::function<void (const Seed::Ptr&)> Callback;

    void drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst);

    void insertFrame(const Frame::Ptr &frame);

    void finishFrame();

    int createSeeds(const KeyFrame::Ptr &kf);

    void enableTrackThread();

    void disableTrackThread();

    void startMainThread();

    void stopMainThread();

    void setMap(const Map::Ptr &map);

    static Ptr create(const FastDetector::Ptr &fast_detector, const Callback &callback, bool report = false, bool verbose = false)
    { return Ptr(new DepthFilter(fast_detector, callback, report, verbose)); }

private:

    DepthFilter(const FastDetector::Ptr &fast_detector, const Callback &callback, bool report, bool verbose);

    Callback seed_coverged_callback_;

    void run();

    void setStop();

    bool isRequiredStop();

    bool checkNewFrame();

    uint64_t trackSeeds();

    int updateSeeds();

    int reprojectSeeds();

    bool earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed);

    bool findEpipolarMatch(const Seed::Ptr &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, Vector2d &px_matched, int &level_matched);

private:

    struct Option{
        int max_kfs; //! max keyframes for seeds tracking(exclude current keyframe)
        double max_epl_length;
        double epl_dist2_threshold;
        double seed_converge_threshold;
        double klt_epslion;
        double align_epslion;
        double min_disparity;
        double min_track_features;
    } options_;

    std::shared_ptr<std::thread> filter_thread_;

    FastDetector::Ptr fast_detector_;

    Map::Ptr map_;

    std::deque<std::pair<Frame::Ptr, bool> > frames_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds> > > seeds_buffer_;
    Seeds tracked_seeds_;

    Frame::Ptr last_frame_;
    Frame::Ptr current_frame_;

    const bool report_;
    const bool verbose_;

    //! main thread
    bool stop_require_;
    std::mutex mutex_stop_;
    std::mutex mutex_frame_;
    //! track thread
    std::condition_variable cond_process_;
    std::future<uint64_t> seeds_track_future_;
    bool track_thread_enabled_;

};

}

#endif //_SSVO_DEPTH_FILTER_HPP_
