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

    void insertNewKeyFrame(const KeyFrame::Ptr &kf, double mean_depth, double min_depth);

private:

    LocalMapper(const FastDetector::Ptr &fast_detector, const Map::Ptr &map, double fps, bool report = false);

    void processNewKeyFrame();

    void createNewMapPoints();

private:

    struct KFD{
        KeyFrame::Ptr kf;
        double mean_depth;
        double min_depth;
        KFD(){}
        KFD(const KeyFrame::Ptr& _kf, double _mean_depth, double _min_depth) :
            kf(_kf), mean_depth(_mean_depth), min_depth(_min_depth){}
    };

    std::shared_ptr<std::thread> mapping_thread_;

    FastDetector::Ptr fast_detector_;

    Map::Ptr map_;

    std::deque<KFD> new_keyframes_;

    KFD keyframe_depth_;
    std::vector<Feature::Ptr> fts_;

    const int delay_;
    const bool report_;

    std::mutex mutex_kfs_;
    std::condition_variable cond_kfs_;

};

}

#endif //_SSVO_LOCAL_MAPPING_HPP_
