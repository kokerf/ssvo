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

    static LocalMapper::Ptr create(const FastDetector::Ptr &fast_detector, const Map::Ptr &map, double fps, bool report = false)
    { return LocalMapper::Ptr(new LocalMapper(fast_detector, map, fps, report));}

    void insertNewKeyFrame(const KeyFrame::Ptr &kf);

private:

    LocalMapper(const FastDetector::Ptr &fast_detector, const Map::Ptr &map, double fps, bool report = false);

    void processKeyFrame();

private:

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    Map::Ptr map_;

    std::deque<KeyFrame::Ptr> new_keyframes_;

    KeyFrame::Ptr current_keyframe_;

    const int delay_;
    const bool report_;

    std::mutex mutex_kfs_;
    std::condition_variable cond_kfs_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
