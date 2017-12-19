#ifndef _FEATURE_TRACKER_HPP_
#define _FEATURE_TRACKER_HPP_
#endif

#include "global.hpp"
#include "feature_detector.hpp"
#include "map.hpp"

namespace ssvo
{

class FeatureTracker : public noncopyable
{
    struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        MapPoint::Ptr pt;    //!< 3D point.
        Vector2d px;         //!< projected 2D pixel location.
        Candidate(MapPoint::Ptr pt, Vector2d& px) : pt(pt), px(px) {}
        Candidate(){}
    };

    struct Grid {
        typedef std::list<Candidate, aligned_allocator<Candidate> > Cell;
        int grid_size;
        int grid_n_cols;
        int grid_n_rows;
        std::vector<Cell*> cells;
        std::vector<int> grid_order;
    };

public:
    typedef std::shared_ptr<FeatureTracker> Ptr;

    ~FeatureTracker();

    int reprojectLoaclMap(const Frame::Ptr &frame, const Map::Ptr &map);

    inline static FeatureTracker::Ptr create(int width, int height, int grid_size, bool report = false, bool verbose = false)
    {return FeatureTracker::Ptr(new FeatureTracker(width, height, grid_size, report, verbose));}

private:

    FeatureTracker(int width, int height, int grid_size, bool report = false, bool verbose = false);

    void resetGrid();

    bool reprojectMapPoint(const Frame::Ptr &frame, const MapPoint::Ptr &point);

    bool trackMapPoints(const Frame::Ptr &frame, Grid::Cell &cell);

private:

    struct Option{
        int border;
        int max_kfs;
    } options_;

    Grid grid_;

    bool report_;
    bool verbose_;
};

namespace utils{

void getWarpMatrixAffine(
    const Camera::Ptr &cam_ref,
    const Camera::Ptr &cam_cur,
    const Vector2d &px_ref,
    const Vector3d &f_ref,
    const int level_ref,
    const double depth_ref,
    const Sophus::SE3d &T_cur_ref,
    const int patch_size,
    Matrix2d &A_cur_ref);

template<typename Td, int size>
void warpAffine(
    const cv::Mat &img_ref,
    Matrix<Td, size, size, RowMajor> &patch,
    const Matrix2d &A_cur_from_ref,
    const Vector2d &px_ref,
    const int level_ref,
    const int level_cur);

}

}//! end of ssvo