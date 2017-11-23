#ifndef _MAP_HPP_
#define _MAP_HPP_

#include "map_point.hpp"
#include "keyframe.hpp"
#include "global.hpp"

namespace ssvo{

class Map: public noncopyable
{
public:
    typedef std::shared_ptr<Map> Ptr;

    void clear();

    void insertKeyFrame(const KeyFrame::Ptr kf);

    void removeKeyFrame(const KeyFrame::Ptr kf);

    void insertMapPoint(const MapPoint::Ptr mpt);

    void removeMapPoint(const MapPoint::Ptr mpt);

    std::vector<KeyFrame::Ptr> getAllKeyFrames();

    std::vector<KeyFrame::Ptr> getAllKeyFramesOrderedByID();

    std::vector<MapPoint::Ptr> getAllMapPoints();

    uint64_t KeyFramesInMap();

    uint64_t MapPointsInMap();

    inline static Map::Ptr create() {return Map::Ptr(new Map());}

private:

    std::unordered_set<KeyFrame::Ptr> kfs_;

    std::unordered_set<MapPoint::Ptr> mpts_;

    std::mutex mutex_kf_;
    std::mutex mutex_mpt_;
};

}

#endif