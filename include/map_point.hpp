#ifndef _MAP_POINT_HPP_
#define _MAP_POINT_HPP_

#include <map>
#include <Eigen/Dense>

#include "global.hpp"

class MapPoint
{
public:
    MapPoint(Vector3d& p, FramePtr& frame);

public:

    static long int map_point_id_;
    const long int id_;

    Vector3d pos_;

};


#endif