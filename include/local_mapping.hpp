#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include "global.hpp"
#include "map.hpp"
#include "feature_detector.hpp"
#include "alignment.hpp"

namespace ssvo{

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

    bool findEpipolarMatch(const Feature::Ptr &ft, const KeyFrame::Ptr &keyframe, const Frame::Ptr &frame,
                           const SE3d &T_cur_from_ref, const double sigma, double &depth);

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
