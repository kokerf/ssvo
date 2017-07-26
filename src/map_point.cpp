#include "map_point.hpp"

long int MapPoint::map_point_id_ = 0;

MapPoint::MapPoint(Vector3d& p, FramePtr& frame):
    id_(map_point_id_++), pos_{p}
{

}