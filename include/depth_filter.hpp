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

//    typedef std::function<void (const Seed::Ptr&)> Callback;

    void trackFrame(const Frame::Ptr &frame_last, const Frame::Ptr &frame_cur);

    void insertFrame(const Frame::Ptr &frame);

    void insertKeyFrame(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame);

//    int getSeedsForMapping(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame);

    bool perprocessSeeds(const KeyFrame::Ptr &keyframe);

    void enableTrackThread();

    void disableTrackThread();

    void startMainThread();

    void stopMainThread();

    void logSeedsInfo();

    static Ptr create(const FastDetector::Ptr &fast_detector, const LocalMapper::Ptr &mapper, bool report = false, bool verbose = false)
    { return Ptr(new DepthFilter(fast_detector, mapper, report, verbose)); }

private:

    DepthFilter(const FastDetector::Ptr &fast_detector, const LocalMapper::Ptr &mapper, bool report, bool verbose);

//    SeedCallback seed_coverged_callback_;

    void run();

    void setStop();

    bool isRequiredStop();

    Frame::Ptr checkNewFrame();

    int createSeeds(const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame = nullptr);

    int trackSeeds(const Frame::Ptr &frame_last, const Frame::Ptr &frame_cur) const;

    int updateSeeds(const Frame::Ptr &frame);

    int reprojectAllSeeds(const Frame::Ptr &frame);

    int reprojectSeeds(const KeyFrame::Ptr& keyframe, Seeds& seeds, const Frame::Ptr &frame);

    void createFeatureFromSeed(const Seed::Ptr &seed);

    bool earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed);

    bool findEpipolarMatch(const Seed::Ptr &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, Vector2d &px_matched, int &level_matched);

private:

    struct Option{
        int max_kfs; //! max keyframes for seeds tracking(exclude current keyframe)
        int max_seeds_buffer;
        int max_features;
        double max_perprocess_kfs;
        double max_epl_length;
        double epl_dist2_threshold;
        double klt_epslion;
        double align_epslion;
        double px_error_normlized;
    } options_;

    FastDetector::Ptr fast_detector_;
    LocalMapper::Ptr mapper_;

    std::deque<Frame::Ptr> frames_buffer_;
    std::deque<Frame::Ptr> passed_frames_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds> > > seeds_buffer_;
    std::map<uint64_t, std::tuple<int, int> > seeds_convergence_rate_;

    const bool report_;
    const bool verbose_;

    //! main thread
    std::shared_ptr<std::thread> filter_thread_;

    bool stop_require_;
    std::mutex mutex_stop_;
    std::mutex mutex_frame_;
    std::mutex mutex_seeds_;
    //! track thread
    std::condition_variable cond_process_main_;
    std::condition_variable cond_process_sub_;
    std::future<int> seeds_track_future_;
    bool track_thread_enabled_;

};

}

#endif //_SSVO_DEPTH_FILTER_HPP_
