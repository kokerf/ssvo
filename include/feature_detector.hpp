#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include "fast/fast.h"
#include "global.hpp"

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
    Corner() {}
    Corner(int x, int y, float score, int level) : x(x), y(y), level(level), score(score) {}

    Corner(const Corner& other): x(other.x), y(other.y), level(other.level), score(other.score) {}
};

typedef std::vector<Corner> Corners;

class Grid
{
public:
    Grid(int cols, int rows, int grid_size, int grid_min_size);

    inline const int getSize() const {return grid_size_; }

    inline const int getGridIndex(const int x, const int y) const {return y/grid_size_*grid_n_cols_ + x/grid_size_; }

    inline const bool getOccupancy(const int n) const { return occupancy_[n]; }

    inline const bool getOccupancy(const int x, const int y) const { return getOccupancy(getGridIndex(x, y)); }

    const int getCorners(std::vector<Corner> &corners) const;

    void resetOccupancy();

    void resetSize(int grid_size);

    bool setOccupancy(const Corner &corner);

    const int setOccupancy(const std::vector<Corner> &corners);

    const int setOccupancyAdaptive(const std::vector<Corner> &corners, const int N);


private:
    const int cols_;
    const int rows_;
    int grid_size_;
    const int grid_min_size_;
    int grid_n_cols_;
    int grid_n_rows_;
    std::vector<bool> occupancy_;
    std::vector<Corner> corners_;
};

class FastDetector: public noncopyable
{
public:

    typedef std::shared_ptr<FastDetector> Ptr;

    int detect(const ImgPyr& img_pyr, std::vector<Corner>& corners, const std::vector<Corner>& exist_corners, const int N, const double eigen_threshold = 30.0);

    void drawGrid(const cv::Mat& img, cv::Mat& img_grid);

    inline static FastDetector::Ptr create(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold = 20, int min_threshold = 7)
    {return FastDetector::Ptr(new FastDetector(width, height, border, nlevels, grid_size, grid_min_size, max_threshold, min_threshold));}

private:
    FastDetector(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold, int min_threshold);

    int detectInLevel(const cv::Mat& img, int level, const double eigen_threshold = 30.0);

    float shiTomasiScore(const cv::Mat& img, int u, int v);

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

    std::vector<std::vector<Corner> > corners_in_levels_;
    Grid grid_fliter_;
};

}//! end of ssvo

#endif