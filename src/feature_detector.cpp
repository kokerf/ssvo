#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "feature_detector.hpp"

namespace ssvo{

const bool operator < (const Corner &a, const Corner &b)
{
    return a.score < b.score;
}

const bool operator == (const Corner &a, const Corner &b)
{
    return a.x == b.x && a.y == b.y;
}

template <>
inline size_t Grid<Corner>::getIndex(const Corner &element)
{
    return static_cast<size_t>(element.y/grid_size_)*grid_n_cols_
        + static_cast<size_t>(element.x/grid_size_);
}

//! FastDetector
FastDetector::FastDetector(int width, int height, int border, int nlevels,
                           int grid_size, int grid_min_size, int max_threshold, int min_threshold):
    width_(width), height_(height), border_(border), nlevels_(nlevels), grid_min_size_(grid_min_size),
    size_adjust_(grid_size!=grid_min_size), max_threshold_(max_threshold), min_threshold_(min_threshold),
    threshold_(max_threshold_), grid_filter_(width, height, grid_size)
{
    corners_in_levels_.resize(nlevels_);
}

int FastDetector::detect(const ImgPyr &img_pyr, Corners &new_corners, const Corners &exist_corners,
                         const int N, const double eigen_threshold)
{
    LOG_ASSERT(img_pyr.size() == nlevels_) << "Unmatch size of ImgPyr(" << img_pyr.size() << ") with nlevel(" << nlevels_ << ")";
    LOG_ASSERT(img_pyr[0].size() == cv::Size(width_, height_)) << "Error cv::Mat size: " << img_pyr[0].size();

    //! 1. Corners detect in all levels
    for(Corners &cs : corners_in_levels_) { cs.clear(); }
    //! find a good threshold to detect fast
    detectAdaptive(img_pyr[0], corners_in_levels_[0], 1.5*N, eigen_threshold, 5);
    
    int new_coners = corners_in_levels_[0].size();
    for(int level = 1; level < nlevels_; level++)
    {
        new_coners += detectInLevel(img_pyr[level], corners_in_levels_[level], level, max_threshold_, eigen_threshold, border_);
    }

    //! 2. Get corners from grid
    setCorners(grid_filter_, exist_corners);
//    setGridMask(grid_filter_, exist_corners);

    for(const Corners &corners : corners_in_levels_)
        setCorners(grid_filter_, corners);

    //! if adjust the grid size
    if(size_adjust_)
    {
        resetGridAdaptive(grid_filter_, N, grid_min_size_);
    }

    setGridMask(grid_filter_, exist_corners);
    grid_filter_.getBestElement(new_corners);
    grid_filter_.clear();

    return new_corners.size();
}

void FastDetector::setGridMask(Grid<Corner> &grid, const Corners &corners)
{
    for(const Corner &corner : corners)
    {
        const size_t id = grid.getIndex(corner);
        grid.setMask(id);
    }
}

void FastDetector::setCorners(Grid<Corner> &grid, const Corners &corners)
{
    for(const Corner &corner : corners)
        grid.insert(corner);
}

//! detect in level 0 to find a good threshold
void FastDetector::detectAdaptive(const cv::Mat &img, Corners &corners, const size_t required, const double eigen_threshold, const int trials)
{
    const size_t min_corners = required >> 1;

    int threshold = threshold_;
    const int adjust = 1;
    for(int i = 0; i < trials && threshold <= max_threshold_ && threshold_ >= min_threshold_; i++)
    {
        size_t num = detectInLevel(img, corners, 0, threshold_, eigen_threshold, border_);

        if(num < 0.8 * min_corners)
        {
            threshold -= adjust;
        }
        else if(num > 1.5 * min_corners)
        {
            threshold += adjust;
            break;
        }
    }

    threshold_ = MAX(MIN(threshold, max_threshold_), min_threshold_);
}

size_t FastDetector::detectInLevel(const cv::Mat &img,
                                   Corners &corners,
                                   const int level,
                                   const int threshold,
                                   const double eigen_threshold,
                                   const int border)
{
    LOG_ASSERT(img.type() == CV_8UC1) << "Error cv::Mat type: " << img.type();

    const int rows = img.rows;
    const int cols = img.cols;
    const int stride = cols;
    const int scale = 1 << level;
    const int max_cols = cols-border;
    const int max_rows = rows-border;

    std::vector<fast::fast_xy> fast_corners;

#if __SSE2__
    fast::fast_corner_detect_10_sse2(img.data, cols, rows, stride, threshold, fast_corners);
#elif HAVE_FAST_NEON
    fast::fast_corner_detect_9_neon(img.data, cols, rows, stride, threshold, fast_corners);
#else
    fast::fast_corner_detect_10(img.data, cols, rows, stride, threshold, fast_corners);
#endif

    std::vector<int> scores, nm_corners;
    fast::fast_corner_score_10(img.data, stride, fast_corners, threshold, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    corners.clear();
    corners.reserve(fast_corners.size());
    for(int& index : nm_corners)
    {
        fast::fast_xy& xy = fast_corners[index];
        const int u = xy.x;
        const int v = xy.y;

        //! border check;
        if(u < border || v < border || u > max_cols || v > max_rows)
            continue;

        const float score = shiTomasiScore(img, u, v);

        //! reject the low-score point
        if(score < eigen_threshold)
            continue;

        const int x = u*scale;
        const int y = v*scale;

        corners.emplace_back(Corner(x, y, score, level));
    }

//    fast_corners.clear();

    return corners.size();
}


void FastDetector::drawGrid(const cv::Mat& img, cv::Mat& img_grid)
{
    img_grid = img.clone();

    int grid_size = grid_filter_.gridSize();
    int grid_n_cols = (ceil(static_cast<double>(width_)/grid_size));
    int grid_n_rows = (ceil(static_cast<double>(height_)/grid_size));

    for(int c = 1; c < grid_n_cols; c++)
    {
        cv::Point start(grid_size*c-1, 0);
        cv::Point end(grid_size*c-1, img.rows-1);
        cv::line(img_grid, start, end, cv::Scalar(0, 50, 0));
    }

    for(int l = 1; l < grid_n_rows; l++)
    {
        cv::Point start(0, grid_size*l-1);
        cv::Point end(img.cols-1, grid_size*l-1);
        cv::line(img_grid, start, end, cv::Scalar(0, 50, 0));
    }
}

//! site from rpg_vikit
//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/src/vision.cpp#L113
//! opencv ref: https://github.com/opencv/opencv/blob/26be2402a3ad6c9eacf7ba7ab2dfb111206fffbb/modules/imgproc/src/corner.cpp#L129
inline float FastDetector::shiTomasiScore(const cv::Mat& img, int u, int v)
{
    LOG_ASSERT(img.type() == CV_8UC1) << "Error cv::Mat type:" << img.type();

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2*halfbox_size;
    const int box_area = box_size*box_size;
    const int x_min = u-halfbox_size;
    const int x_max = u+halfbox_size;
    const int y_min = v-halfbox_size;
    const int y_max = v+halfbox_size;

    if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for( int y=y_min; y<y_max; ++y )
    {
        const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
        {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx*dx;
            dYY += dy*dy;
            dXY += dx*dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - std::sqrt( (dXX - dYY) * (dXX - dYY) +  dXY * dXY));
}

}