#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include "fast/fast.h"
#include "global.hpp"
#include "grid.hpp"

//! (u,v) is in the n-level image of pyramid
//! (x,y) is in the 0-level image of pyramid
namespace ssvo
{

struct Corner{
    float x;        //!< x-coordinate of corner in the image.
    float y;        //!< y-coordinate of corner in the image.
    int level;      //!< pyramid level of the corner.
    float score;    //!< shi-tomasi score of the corner.
    //float angle;  //!< for gradient-features: dominant gradient angle.
    Corner() : x(-1), y(-1), level(-1), score(-1) {}
    Corner(int x, int y, float score, int level) : x(x), y(y), level(level), score(score) {}

    Corner(const Corner& other): x(other.x), y(other.y), level(other.level), score(other.score) {}
};

typedef std::vector<Corner> Corners;

class FastDetector: public noncopyable
{
public:

    typedef std::shared_ptr<FastDetector> Ptr;

    int detect(const ImgPyr &img_pyr, Corners &new_corners, const Corners &exist_corners, const int N, const double eigen_threshold = 30.0);

    void drawGrid(const cv::Mat &img, cv::Mat &img_grid);

    static float shiTomasiScore(const cv::Mat &img, int u, int v);

    static size_t detectInLevel(const cv::Mat &img, Corners &corners, const int level, const int threshold, const double eigen_threshold=30, const int border=4);

    inline static FastDetector::Ptr create(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold = 20, int min_threshold = 7)
    {return FastDetector::Ptr(new FastDetector(width, height, border, nlevels, grid_size, grid_min_size, max_threshold, min_threshold));}

private:

    FastDetector(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold, int min_threshold);

    static void setGridMask(Grid<Corner> &grid, const Corners &corners);

    static void setCorners(Grid<Corner> &grid, const Corners &corners);

    void detectAdaptive(const cv::Mat &img, Corners &corners, const size_t required, const double eigen_threshold, const int trials = 5);

private:

    const int width_;
    const int height_;
    const int border_;
    const int nlevels_;
    int N_;

    const int grid_min_size_;
    bool size_adjust_;
    const int max_threshold_;
    const int min_threshold_;
    int threshold_;

    std::vector<Corners> corners_in_levels_;
    Grid<Corner> grid_filter_;
};

}//! end of ssvo

#endif