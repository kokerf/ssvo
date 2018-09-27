#ifndef _SSVO_DEPTH_FILTER_HPP_
#define _SSVO_DEPTH_FILTER_HPP_

#include "global.hpp"
#include "map.hpp"
#include "seed.hpp"
#include "feature_detector.hpp"
#include "local_mapping.hpp"

namespace ssvo
{

class DepthFilter : public noncopyable
{
public:

    typedef std::shared_ptr<DepthFilter> Ptr;

    typedef std::function<void (const Seed::Ptr&)> SeedCallback;
    typedef std::function<void (const KeyFrame::Ptr&)> KeyFrameCallback;

    void insertFrame(const Frame::Ptr &frame, const KeyFrame::Ptr keyframe = nullptr);

    void insertKeyFrame(const KeyFrame::Ptr &keyframe);

    void startMainThread();

    void stopMainThread();

    void logSeedsInfo();

    void setSeedConvergedCallback(const SeedCallback &callback);

    void setKeyFrameProcessCallback(const KeyFrameCallback &callback);

    static Ptr create(const FastDetector::Ptr &fast_detector, bool report = false, bool verbose = false)
    { return Ptr(new DepthFilter(fast_detector, report, verbose)); }

private:

    DepthFilter(const FastDetector::Ptr &fast_detector, bool report, bool verbose);

    SeedCallback seed_converged_callback_;

    KeyFrameCallback keyframe_process_callback_;

    void run();

    void setStop();

    bool isRequiredStop();

    bool checkNewFrame(Frame::Ptr &frame, KeyFrame::Ptr &keyframe);

    int createSeeds(const KeyFrame::Ptr &keyframe);

    int updateByConnectedKeyFrames(const KeyFrame::Ptr &keyframe, int num = 2);

    int reprojectAllSeeds(const Frame::Ptr &frame);

    int reprojectSeeds(const KeyFrame::Ptr& keyframe, const Frame::Ptr &frame, double epl_err, double px_error, bool created = true);

    bool findEpipolarMatch(const Seed::Ptr &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, Vector2d &px_matched, int &level_matched);

private:

    struct Option{
        int max_kfs; //! max keyframes for seeds tracking(exclude current keyframe)
        int max_features;
        double max_perprocess_kfs;
        double max_epl_length;
        double epl_dist2_threshold;
        double klt_epslion;
        double align_epslion;
        double pixel_error_threshold;
        double min_frame_disparity;
        double min_pixel_disparity;
    } options_;

    FastDetector::Ptr fast_detector_;

    std::deque<std::pair<Frame::Ptr, KeyFrame::Ptr> > frames_buffer_;
//    std::map<uint64_t, std::tuple<int, int> > seeds_convergence_rate_;

    const bool report_;
    const bool verbose_;

    //! main thread
    std::shared_ptr<std::thread> filter_thread_;

    bool stop_require_;
    std::mutex mutex_stop_;
    std::mutex mutex_frame_;
    //! track thread
    std::condition_variable cond_process_main_;
    std::future<int> seeds_track_future_;
};

}

#endif //_SSVO_DEPTH_FILTER_HPP_
