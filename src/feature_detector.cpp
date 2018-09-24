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

FastGrid::FastGrid(int width, int height, int cell_size, int max_threshold, int min_threshold) :
    width_(width), height_(height), max_threshold_(max_threshold), min_threshold_(min_threshold), cell_size_(cell_size)
{
    cell_size_ = MIN(MIN(width_, height_), cell_size_);
    if(cell_size_ < MIN_CEIL_SIZE) cell_size_ = MIN_CEIL_SIZE;

    int min_size = std::floor(std::sqrt(static_cast<float>(width_*height_)/MAX_GRIDS));

    if(cell_size_ < min_size) cell_size_ = min_size;

    cell_n_cols_ = std::floor(static_cast<float>(width_)/cell_size_);
    cell_n_rows_ = std::floor(static_cast<float>(height_)/cell_size_);

    N_ = cell_n_cols_ * cell_n_rows_;
    cells_x_.resize(cell_n_cols_+1, 0);
    cells_y_.resize(cell_n_rows_+1, 0);
    fast_threshold_.resize(N_, max_threshold_);

    for (auto itr = cells_x_.begin() + 1; itr != cells_x_.end(); itr++)
        *itr = *(itr - 1) + cell_size_;

    for (auto itr = cells_y_.begin() + 1; itr != cells_y_.end(); itr++)
        *itr = *(itr - 1) + cell_size_;

    cells_x_[cell_n_cols_] = width_;
    cells_y_[cell_n_rows_] = height_;
}

cv::Rect FastGrid::getCell(int id) const
{
    const int r = id / cell_n_cols_;
    const int c = id % cell_n_cols_;

    LOG_ASSERT(r <= cell_n_rows_) << "Out of scope ! r = " << r << " should not  big than << " << cell_n_rows_;

    return cv::Rect(cells_x_[c], cells_y_[r], cells_x_[c+1] - cells_x_[c], cells_y_[r+1] - cells_y_[r]);
}

int FastGrid::getThreshold(int id) const
{
    LOG_ASSERT(id <= N_) << "Out of scope ! id = " << id << " should not big than << " << N_;
    return fast_threshold_.at(id);
}

bool FastGrid::setThreshold(int id, int threshold)
{
    LOG_ASSERT(id <= N_) << "Out of scope ! id = " << id << " should not big than << " << N_;
    if(threshold > max_threshold_) threshold = max_threshold_;
    if(threshold < min_threshold_) threshold = min_threshold_;
    fast_threshold_.at(id) = threshold;
    return true;
}

bool FastGrid::inBoundary(int id) const
{
    const int r = id / cell_n_cols_;
    const int c = id % cell_n_cols_;

    return (r == 0 || c == 0 || r == cell_n_rows_ || c == cell_n_cols_);
}

//! FastDetector
FastDetector::FastDetector(int width, int height, int border, int nlevels, double scale,
                           int grid_size, int grid_min_size, int max_threshold, int min_threshold):
    width_(width), height_(height), border_(border), nlevels_(nlevels),
    scale_factor_(scale), log_scale_factor_(log(scale)), grid_min_size_(grid_min_size),
    size_adjust_(grid_size!=grid_min_size), max_threshold_(max_threshold), min_threshold_(min_threshold),
    threshold_(max_threshold_), grid_filter_(width, height, grid_size)
{
    scale_factors_.resize(nlevels_, 1.0);
    inv_scale_factors_.resize(nlevels_, 1.0);
    for(int i = 1; i < nlevels_; i++)
    {
        scale_factors_[i] = scale_factors_[i-1] * scale_factor_;
        inv_scale_factors_[i] = 1.0f / scale_factors_[i];
    }

    level_sigma2_.resize(nlevels_, 1.0);
    inv_level_sigma2_.resize(nlevels_, 1.0);
    for(int i = 1; i < nlevels_; i++)
    {
        level_sigma2_[i] = scale_factors_[i] * scale_factors_[i];
        inv_level_sigma2_[i] = 1.0f / level_sigma2_[i];
    }

    corners_in_levels_.resize(nlevels_);
    for(int i = 0; i < nlevels_; ++i)
    {
        detect_grids_.push_back(FastGrid(std::round(width_*inv_scale_factors_[i]), std::round(height_*inv_scale_factors_[i]), grid_size, max_threshold_, min_threshold_));
    }
}

int FastDetector::getNLevels() const
{
    return nlevels_;
}

double FastDetector::getScaleFactor() const
{
    return scale_factor_;
}

double FastDetector::getLogScaleFactor() const
{
    return log_scale_factor_;
}

std::vector<double> FastDetector::getScaleFactors() const
{
    return scale_factors_;
}

std::vector<double> FastDetector::getInvScaleFactors() const
{
    return inv_scale_factors_;
}

std::vector<double> FastDetector::getLevelSigma2() const
{
    return level_sigma2_;
}

std::vector<double> FastDetector::getInvLevelSigma2() const
{
    return inv_level_sigma2_;
}

size_t FastDetector::detect(const ImgPyr &img_pyr, Corners &new_corners, const Corners &exist_corners,
                         const int N, const double eigen_threshold)
{
    LOG_ASSERT(img_pyr.size() == nlevels_) << "Unmatch size of ImgPyr(" << img_pyr.size() << ") with nlevel(" << nlevels_ << ")";
    LOG_ASSERT(img_pyr[0].size() == cv::Size(width_, height_)) << "Error cv::Mat size: " << img_pyr[0].size();

    //! 1. Corners detect in all levels
    for(Corners &cs : corners_in_levels_) { cs.clear(); }
    
    size_t new_coners = corners_in_levels_[0].size();
    for(int level = 0; level < nlevels_; level++)
    {
        new_coners += detectInLevel(img_pyr[level], detect_grids_[level], corners_in_levels_[level], eigen_threshold, border_);

        const double scale = scale_factors_.at(level);
        for(Corner &corner : corners_in_levels_[level])
        {
            corner.level = level;
            corner.x *= scale;
            corner.y *= scale;
        }
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

size_t FastDetector::detectInLevel(const cv::Mat &img,
                                   FastGrid &fast_grid,
                                   Corners &corners,
                                   const double eigen_threshold,
                                   const int border)
{
    LOG_ASSERT(img.type() == CV_8UC1) << "Error cv::Mat type: " << img.type();
    const int rows = img.rows;
    const int cols = img.cols;

    LOG_ASSERT(fast_grid.width_ == cols && fast_grid.height_ == rows) << "The grid(" << fast_grid.width_ << "*" << fast_grid.height_ << ") is not fit the image("<< cols << "*" << rows << ")";

    const int max_cols = cols-border;
    const int max_rows = rows-border;

    corners.clear();
    static float corner_density = 1.0f / (20*20);
    for(int i = 0; i < fast_grid.nCells(); ++i)
    {
        Corners corners_per_cell;
        const cv::Rect rect = fast_grid.getCell(i);
        const int th = fast_grid.getThreshold(i);
        //! fast detect
        fastDetect(img(rect), corners_per_cell, th, eigen_threshold);
        //! fast re-detect
        if(corners_per_cell.empty() && th != fast_grid.min_threshold_)
        {
            fastDetect(img(rect), corners_per_cell, fast_grid.min_threshold_, eigen_threshold);
            fast_grid.setThreshold(i, fast_grid.min_threshold_);
        }
        else if(static_cast<float>(corners_per_cell.size()) / (rect.width*rect.height) > corner_density)
        {
            fast_grid.setThreshold(i, th+std::ceil((fast_grid.max_threshold_-fast_grid.min_threshold_)*0.1 + fast_grid.min_threshold_));
        }

        corners.reserve(corners.size() + corners_per_cell.size());

        const bool border_check = fast_grid.inBoundary(i);
        if(border_check)
        {
            for(Corner &corner : corners_per_cell)
            {
                corner.x += rect.x;
                corner.y += rect.y;
                //! border check;
                if(corner.x < border || corner.y < border || corner.x > max_cols || corner.y > max_rows)
                    continue;

                corners.push_back(corner);
            }
        }
        else
        {
            for(Corner &corner : corners_per_cell)
            {
                corner.x += rect.x;
                corner.y += rect.y;
                corners.push_back(corner);
            }
        }
    }

    return corners.size();
}

void FastDetector::fastDetect(const cv::Mat &img, Corners &corners, int threshold, double eigen_threshold)
{
    int cols = img.cols;
    int rows = img.rows;
    int stride = img.step.p[0];

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

        const float score = shiTomasiScore(img, u, v);

        //! reject the low-score point
        if(score < eigen_threshold)
            continue;

        corners.emplace_back(Corner(u, v, score, -1));
    }
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