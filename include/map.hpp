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

    inline uint64_t KeyFramesInMap() {return kfs_.size();}

    uint64_t MapPointsInMap() {return mpts_.size();}

    inline static Map::Ptr create() {return std::make_shared<Map>();}

private:

    std::unordered_set<KeyFrame::Ptr> kfs_;

    std::unordered_set<MapPoint::Ptr> mpts_;
};

}

#endif