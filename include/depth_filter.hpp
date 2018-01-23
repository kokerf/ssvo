#ifndef _SSVO_DEPTH_FILTER_HPP_
#define _SSVO_DEPTH_FILTER_HPP_

#include "global.hpp"
#include "map.hpp"
#include "seed.hpp"
#include "feature_detector.hpp"

namespace ssvo
{

class DepthFilter : public noncopyable
{
public:

    typedef std::shared_ptr<DepthFilter> Ptr;

    typedef std::function<void (const Seed::Ptr&)> Callback;

    void trackFrame(const Frame::Ptr &frame_last, const Frame::Ptr &frame_cur);

    void insertFrame(const Frame::Ptr &frame);

    int createSeeds(const KeyFrame::Ptr &kf, const Frame::Ptr &frame = nullptr);

    void enableTrackThread();

    void disableTrackThread();

    void startMainThread();

    void stopMainThread();

    static Ptr create(const FastDetector::Ptr &fast_detector, const Callback &callback, bool report = false, bool verbose = false)
    { return Ptr(new DepthFilter(fast_detector, callback, report, verbose)); }

private:

    DepthFilter(const FastDetector::Ptr &fast_detector, const Callback &callback, bool report, bool verbose);

    Callback seed_coverged_callback_;

    void run();

    void setStop();

    bool isRequiredStop();

    Frame::Ptr checkNewFrame();

    int trackSeeds(const Frame::Ptr &frame_last, const Frame::Ptr &frame_cur) const;

    int updateSeeds(const Frame::Ptr &frame);

    int reprojectSeeds(const Frame::Ptr &frame);

    bool earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed);

    bool findEpipolarMatch(const Seed::Ptr &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, Vector2d &px_matched, int &level_matched);

private:

    struct Option{
        int max_kfs; //! max keyframes for seeds tracking(exclude current keyframe)
        int max_seeds_buffer;
        double max_epl_length;
        double epl_dist2_threshold;
        double seed_converge_threshold;
        double klt_epslion;
        double align_epslion;
        double min_disparity;
        double min_track_features;
    } options_;

    FastDetector::Ptr fast_detector_;

    Map::Ptr map_;

    std::deque<Frame::Ptr> frames_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds> > > seeds_buffer_;
    Seeds tracked_seeds_;

    const bool report_;
    const bool verbose_;

    //! main thread
    std::shared_ptr<std::thread> filter_thread_;

    bool stop_require_;
    std::mutex mutex_stop_;
    std::mutex mutex_frame_;
    //! track thread
    std::condition_variable cond_process_;
    std::future<int> seeds_track_future_;
    bool track_thread_enabled_;

};

}

#endif //_SSVO_DEPTH_FILTER_HPP_
