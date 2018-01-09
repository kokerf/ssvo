#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include <future>
#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"

namespace ssvo{

class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void setThread(bool enable_main = false, bool enable_track = true);

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points);

    void insertFrame(const Frame::Ptr &frame);

    bool finishFrame();

    KeyFrame::Ptr getReferenceKeyFrame();
//    void insertNewKeyFrame(const KeyFrame::Ptr &keyframe, double mean_depth, double min_depth, bool optimal = true, bool create_seeds = true);

    void drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst);

    static LocalMapper::Ptr create(const FastDetector::Ptr &fast_detector, double fps, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(fast_detector, fps, report, verbose));}

private:

    LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report, bool verbose);

    void run();

    void setStop();

    bool isRequiredStop();

    uint64_t trackSeeds();

    int updateSeeds();

    int reprojectSeeds();

    int createSeeds(bool is_track = true);

    int createNewFeatures();

    bool checkNewFrame();

    bool earseSeed(const KeyFrame::Ptr &keyframe, const Seed::Ptr &seed);

    bool createFeatureFromSeed(const Seed::Ptr &seed);

    bool needCreateKeyFrame();

    void processNewKeyFrame();

    void insertKeyFrame(const KeyFrame::Ptr &keyframe);

    void checkCulling();

    bool findEpipolarMatch(const Seed::Ptr &seed, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, Vector2d &px_matched, int &level_matched);

public:

    Map::Ptr map_;

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

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    std::deque<std::pair<Frame::Ptr, KeyFrame::Ptr> > frames_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds> > > seeds_buffer_;
    Seeds tracked_seeds_;

    Frame::Ptr last_frame_;
    Frame::Ptr current_frame_;
    KeyFrame::Ptr current_keyframe_;

    const int delay_;
    const bool report_;
    const bool verbose_;

    bool stop_require_;
    std::mutex mutex_frame_;
    std::mutex mutex_keyframe_;
    std::mutex mutex_stop_;
    std::condition_variable cond_process_;
    std::future<uint64_t> frame_process_future_;
    bool status_track_thread_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
