#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include <future>
#include "global.hpp"
#include "map.hpp"
#include "depth_filter.hpp"

namespace ssvo{

class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur);

    void insertKeyFrame(const KeyFrame::Ptr &keyframe);

    void createFeatureFromSeed(const Seed::Ptr &seed);

    static LocalMapper::Ptr create(double fps, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(fps, report, verbose));}

private:

    LocalMapper(double fps, bool report, bool verbose);

    int createFeatureFromLocalMap();

    void checkCulling();

public:

    Map::Ptr map_;

private:

    struct Option{
        double min_disparity;
        int min_redundant_observations;
    } options_;

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    std::deque<std::pair<Frame::Ptr, KeyFrame::Ptr> > frames_buffer_;
    std::deque<std::pair<KeyFrame::Ptr, std::shared_ptr<Seeds> > > seeds_buffer_;
    Seeds tracked_seeds_;

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
