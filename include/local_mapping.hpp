#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include <future>
#include "global.hpp"
#include "map.hpp"

namespace ssvo{

class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur);

    void insertKeyFrame(const KeyFrame::Ptr &keyframe);

    void startMainThread();

    void stopMainThread();

    void addOptimalizeMapPoint(const MapPoint::Ptr &mpt);

    void refineMapPoints(const int max_optimalize_num = -1);

    static LocalMapper::Ptr create(double fps, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(fps, report, verbose));}

private:

    LocalMapper(double fps, bool report, bool verbose);

    void run();

    void setStop();

    bool isRequiredStop();

    KeyFrame::Ptr checkNewKeyFrame();

    int createFeatureFromSeedFeature(const KeyFrame::Ptr &keyframe);

    int createFeatureFromLocalMap(const KeyFrame::Ptr &keyframe);

    void checkCulling(const KeyFrame::Ptr &keyframe);

public:

    Map::Ptr map_;

private:

    struct Option{
        double min_disparity;
        int min_redundant_observations;
        bool enable_local_ba;
        int num_loacl_ba_kfs;
    } options_;

    FastDetector::Ptr fast_detector_;

    std::deque<KeyFrame::Ptr> keyframes_buffer_;

    const int delay_;
    const bool report_;
    const bool verbose_;

    std::shared_ptr<std::thread> mapping_thread_;

    std::list<MapPoint::Ptr> optimalize_candidate_mpts_;

    bool stop_require_;
    std::mutex mutex_stop_;
    std::mutex mutex_keyframe_;
    std::mutex mutex_optimalize_mpts_;
    std::condition_variable cond_process_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
