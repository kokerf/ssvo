#ifndef _MAP_HPP_
#define _MAP_HPP_

#include "map_point.hpp"
#include "keyframe.hpp"
#include "global.hpp"

namespace ssvo{

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;

    void insertKeyFrame(const KeyFrame::Ptr kf);

    void deleteKeyFrame(const KeyFrame::Ptr kf);

    void insertMapPoint(const MapPoint::Ptr mpt);

    void deleteMapPoint(const MapPoint::Ptr mpt);

    Map &operator=(const Map&) = delete; //! copy denied

public:

    std::unordered_map<uint64_t, KeyFrame::Ptr> kfs_;

    std::unordered_map<uint64_t, MapPoint::Ptr> mpts_;
};

}

#endif