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
public:
    typedef std::shared_ptr<FeatureTracker> Ptr;

    int reprojectLoaclMap(const Frame::Ptr &frame);

    static int reprojectMapPoint(const Frame::Ptr &frame, const MapPoint::Ptr& mpt, Vector2d &px_cur, int &level_cur,
                                  const int max_iterations = 30, const double epslion = 0.01, const double threshold = 4.0, bool verbose = false);

    static bool trackFeature(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const Feature::Ptr &ft_ref, const MapPoint::Ptr &mpt,
                             Vector2d &px_cur, int &level_cur, const int max_iterations = 30, const double epslion = 0.01, const double threshold = 4.0, bool verbose = false);

    static bool findSubpixelFeature(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const Feature::Ptr &ft_ref, const Vector3d &p3d,
                                    Vector2d &px_cur, const int level_cur, const int max_iterations = 30, const double epslion = 0.01, const double threshold = 4.0, bool verbose = false);

    static int searchBoWForTriangulation(const KeyFrame::Ptr &keyframe1, const KeyFrame::Ptr &keyframe2, std::map<size_t, size_t> &matches, int max_desp, double max_epl_err);

    static int searchBowByProjection(const KeyFrame::Ptr &keyframe, const std::vector<MapPoint::Ptr> &mpts, std::map<MapPoint::Ptr, size_t> &matches, int max_desp, double threshold = 1.0);

    static void drawMatches(const Frame::Ptr frame1, const Frame::Ptr frame2, cv::Mat &out, bool with_proj = false);

    static void drawMatches(const Frame::Ptr frame1, const Frame::Ptr frame2, cv::Mat &out, const std::vector<std::pair<size_t, size_t>> &matches);

    static void drawEplMatch(const cv::Mat &image1, const cv::Mat &image2, cv::Mat &out1, cv::Mat &out2, const Matrix3d &F12, const Vector2d &px1, const Vector2d &px2);

    static void drawAllFeatures(const KeyFrame::Ptr &keyframe, cv::Mat &out);

    static void drawAffine(const cv::Mat &src, cv::Mat &dst, const Vector2d &px_ref, const Matrix2d &A_ref_cur, const int size, const int level);

    inline static FeatureTracker::Ptr create(int width, int height, int grid_size, int border, bool report = false, bool verbose = false)
    {return FeatureTracker::Ptr(new FeatureTracker(width, height, grid_size, border, report, verbose));}

public:

    struct LogInfo {
        int num_frame_matches;
        int num_cells_matches;
        int num_total_project;
        int num_local_mpts;
        double time_frame_match;
        double time_cells_create;
        double time_cells_match;
    } logs_;

private:

    FeatureTracker(int width, int height, int grid_size, int border, bool report = false, bool verbose = false);

    bool reprojectMapPointToCell(const Frame::Ptr &frame, const MapPoint::Ptr &point);

    bool matchMapPointsFromCell(const Frame::Ptr &frame, Grid<std::pair<MapPoint::Ptr, Vector2d>>::Cell &cell);

    int matchMapPointsFromLastFrame(const Frame::Ptr &frame_cur, const Frame::Ptr &frame_last);

private:

    struct Option{
        int border;
        int max_matches;
        int max_track_kfs;
        int num_align_iter;
        double max_align_epsilon;
        double max_align_error2;
    } options_;

    Grid<std::pair<MapPoint::Ptr, Vector2d>> grid_;
    std::vector<size_t> grid_order_;

    bool report_;
    bool verbose_;
};

}//! end of ssvo