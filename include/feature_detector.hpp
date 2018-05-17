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

class FastGrid{
public:

    enum {
        MAX_GRIDS = 600,
        MIN_CEIL_SIZE = 30
    };

    FastGrid(int width, int height, int cell_size, int max_threshold, int min_threshold);

    inline int nCells() const { return N_; }

    inline cv::Rect getCell(int id) const;

    int getThreshold(int id) const;

    bool setThreshold(int id, int threshold);

    bool inBoundary(int id) const;

public:
    const int width_;
    const int height_;
    const int max_threshold_;
    const int min_threshold_;

private:
    int cell_size_;
    int cell_n_cols_;
    int cell_n_rows_;
    int N_;
    std::vector<int> cells_x_;
    std::vector<int> cells_y_;
    std::vector<int> fast_threshold_;
};

typedef std::vector<Corner> Corners;

class FastDetector: public noncopyable
{
public:

    typedef std::shared_ptr<FastDetector> Ptr;

    size_t detect(const ImgPyr &img_pyr, Corners &new_corners, const Corners &exist_corners, const int N, const double eigen_threshold = 30.0);

    void drawGrid(const cv::Mat &img, cv::Mat &img_grid);

    static float shiTomasiScore(const cv::Mat &img, int u, int v);

    static size_t detectInLevel(const cv::Mat &img, FastGrid &fast_grid, Corners &corners, const double eigen_threshold=30, const int border=4);

    static void fastDetect(const cv::Mat &img, Corners &corners, int threshold, double eigen_threshold = 30);

    inline static FastDetector::Ptr create(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold = 20, int min_threshold = 7)
    {return FastDetector::Ptr(new FastDetector(width, height, border, nlevels, grid_size, grid_min_size, max_threshold, min_threshold));}

private:

    FastDetector(int width, int height, int border, int nlevels, int grid_size, int grid_min_size, int max_threshold, int min_threshold);

    static void setGridMask(Grid<Corner> &grid, const Corners &corners);

    static void setCorners(Grid<Corner> &grid, const Corners &corners);

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

    std::vector<FastGrid> detect_grids_;

    std::vector<Corners> corners_in_levels_;
    Grid<Corner> grid_filter_;
};

}//! end of ssvo

#endif