#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include "fast/fast.h"
#include "global.hpp"
#include "feature.hpp"
#include "grid.hpp"

//! (u,v) is in the n-level image of pyramid
//! (x,y) is in the 0-level image of pyramid
namespace ssvo
{

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

class FastDetector: public noncopyable
{
public:

    typedef std::shared_ptr<FastDetector> Ptr;

    size_t detect(const ImgPyr &img_pyr, Corners &new_corners, const Corners &exist_corners, const int N, const double eigen_threshold = 30.0);

    void drawGrid(const cv::Mat &img, cv::Mat &img_grid);

    int getNLevels() const;

    double getScaleFactor() const;

    double getLogScaleFactor() const;

    std::vector<double> getScaleFactors() const;

    std::vector<double> getInvScaleFactors() const;

    std::vector<double> getLevelSigma2() const;

    std::vector<double> getInvLevelSigma2() const;

    static float shiTomasiScore(const cv::Mat &img, int u, int v);

    static size_t detectInLevel(const cv::Mat &img, FastGrid &fast_grid, Corners &corners, const double eigen_threshold=30, const int border=4);

    static void fastDetect(const cv::Mat &img, Corners &corners, int threshold, double eigen_threshold = 30);

    inline static FastDetector::Ptr create(int width, int height, int border, int nlevels, double scale, int grid_size, int grid_min_size, int max_threshold = 20, int min_threshold = 7)
    {return FastDetector::Ptr(new FastDetector(width, height, border, nlevels, scale, grid_size, grid_min_size, max_threshold, min_threshold));}

private:

    FastDetector(int width, int height, int border, int nlevels, double scale, int grid_size, int grid_min_size, int max_threshold, int min_threshold);

    static void setGridMask(Grid<Corner> &grid, const Corners &corners);

    static void setCorners(Grid<Corner> &grid, const Corners &corners);

private:

    const int width_;
    const int height_;
    const int border_;
    const int nlevels_;
    const double scale_factor_;
    const double log_scale_factor_;
    std::vector<double> scale_factors_;
    std::vector<double> inv_scale_factors_;
    std::vector<double> level_sigma2_;
    std::vector<double> inv_level_sigma2_;

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