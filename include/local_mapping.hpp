#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"

namespace ssvo{

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

    std::pair<Frame::Ptr, KeyFrame::Ptr> current_frame;
    std::pair<double, double> current_depth;

    const int delay_;
    const bool report_;

    std::mutex mutex_kfs_;
    std::condition_variable cond_process_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
