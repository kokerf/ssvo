#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "feature_detector.hpp"

namespace ssvo{

//! Grid
Grid::Grid(int cols, int rows, int grid_size, int grid_min_size) :
    cols_(cols), rows_(rows), grid_size_(grid_size), grid_min_size_(grid_min_size),
    grid_n_cols_(ceil(static_cast<double>(cols)/grid_size_)),
    grid_n_rows_(ceil(static_cast<double>(rows)/grid_size_)),
    occupancy_(grid_n_cols_*grid_n_rows_, false),
    corners_(grid_n_cols_*grid_n_rows_, Corner(0,0,0,0))
{}

void Grid::resetOccupancy()
{
    std::fill(occupancy_.begin(), occupancy_.end(), false);
}

inline void Grid::resetSize(int grid_size)
{
    grid_size_ = grid_size;
    grid_n_cols_ = ceil(static_cast<double>(cols_)/grid_size_);
    grid_n_rows_ = ceil(static_cast<double>(rows_)/grid_size_);
    occupancy_.resize(grid_n_cols_*grid_n_rows_);
    corners_.resize(grid_n_cols_*grid_n_rows_);
    resetOccupancy();
}

//! return true if insert to a new grid
inline bool Grid::setOccupancy(const Corner &corner)
{
    const int idx = getGridIndex(corner.x, corner.y);
    const bool occupied = occupancy_.at(idx);

    if(!occupied || (occupied && corners_[idx].score < corner.score)) {
        corners_[idx] = corner;
    }

    occupancy_[idx] = true;
    return !occupied;
}

//! return the number of grids newly inserted a corner
const int Grid::setOccupancy(const std::vector<Corner> &corners)
{
    int now_corners = 0;
    std::for_each(corners.begin(), corners.end(), [&](Corner corner){
        if(setOccupancy(corner))
        {
            now_corners++;
        }
    });
    return now_corners;
}

const int Grid::getCorners(std::vector<Corner> &corners) const
{
    const int grid_size = grid_n_cols_*grid_n_rows_;
    corners.clear();
    corners.reserve(grid_size);
    for(int i = 0; i < grid_size; ++i)
    {
        if(occupancy_[i])
            corners.push_back(corners_[i]);
    }

    return corners.size();
}

const int Grid::setOccupancyAdaptive(const std::vector<Corner> &corners, const int N)
{
    std::vector<Corner> old_corners;
    getCorners(old_corners);

    int new_size = floorf(std::sqrt(cols_ * rows_ / N));

    int n = 0;
    int now_corners = 0;
    while(n++ < 2)
    {
        now_corners = 0;

        resetSize(new_size);

        //! insert old corners first
        now_corners += setOccupancy(old_corners);
        //! then insert new corners
        std::vector<bool> old_corner_occupancy = occupancy_;

        for(std::vector<Corner>::const_iterator it = corners.begin(); it != corners.end(); it++)
        {
            const int idx = getGridIndex(it->x, it->y);
            //! if occupied by old corners, ignore the new corner
            if(old_corner_occupancy.at(idx))
                continue;

            if(setOccupancy(*it))
              now_corners++;
        }

        if(now_corners < 1.1*N && now_corners > 0.9*N)
            break;

        const float corners_per_grid = 1.0 * now_corners / (grid_n_rows_ * grid_n_cols_);
        const float n_grid = N / corners_per_grid;
        new_size = roundf(std::sqrt(cols_ * rows_ / n_grid));
        new_size = MAX(new_size, grid_min_size_);
        LOG_ASSERT(new_size < cols_ || new_size < rows_) << "Error Grid Size: " << new_size;
    }

    return now_corners;
}

//! FastDetector
FastDetector::FastDetector(int width, int height, int border, int nlevels,
                           int grid_size, int grid_min_size, int max_threshold, int min_threshold):
    width_(width), height_(height), border_(border), nlevels_(nlevels), grid_min_size_(grid_min_size),
    size_adjust_(grid_size!=grid_min_size), max_threshold_(max_threshold), min_threshold_(min_threshold),
    grid_fliter_(width, height, grid_size, grid_min_size)
{
    corners_in_levels_.resize(nlevels_);
}

int FastDetector::detect(const ImgPyr& img_pyr, std::vector<Corner>& corners, const std::vector<Corner>& exist_corners,
                         const int N, const double eigen_threshold)
{
    LOG_ASSERT(img_pyr.size() == nlevels_) << "Unmatch size of ImgPyr(" << img_pyr.size() << ") with nlevel(" << nlevels_ << ")";
    LOG_ASSERT(img_pyr[0].size() == cv::Size(width_, height_)) << "Error cv::Mat size: " << img_pyr[0].size();

    grid_fliter_.resetOccupancy();
    grid_fliter_.setOccupancy(exist_corners);

    Grid new_grid_fliter_(width_, height_, grid_fliter_.getSize(), grid_min_size_);

    int new_coners_ = 0;
    for(int level = 0; level < nlevels_; level++)
    {
        detectInLevel(img_pyr[level], level, eigen_threshold);

        new_coners_ += new_grid_fliter_.setOccupancy(corners_in_levels_[level]);
    }

    new_grid_fliter_.getCorners(corners);

    //! if adjust the grid size
    if(size_adjust_)
    {
        grid_fliter_.setOccupancyAdaptive(corners, N);
        grid_fliter_.getCorners(corners);
    }
    else
    {
        if(corners.size() + exist_corners.size() > N)
        {
            std::sort(corners.begin(), corners.end(), [](Corner c1, Corner c2){return c1.score>c2.score;});
            corners.resize(N-exist_corners.size());
        }

        std::for_each(exist_corners.begin(), exist_corners.end(), [&](Corner corner){
            corners.push_back(corner);
        });
    }

    return corners.size();
}

int FastDetector::detectInLevel(const cv::Mat& img, int level, const double eigen_threshold)
{
    LOG_ASSERT(img.type() == CV_8UC1) << "Error cv::Mat type:" << img.type();

    const int rows = img.rows;
    const int cols = img.cols;
    const int stride = cols;
    const int scale = 1 << level;

    std::vector<fast::fast_xy> fast_corners;

    //cv::imshow("img", img);
    //cv::waitKey(0);

#if __SSE2__
    fast::fast_corner_detect_10_sse2(img.data, cols, rows, stride, max_threshold_, fast_corners);
#elif HAVE_FAST_NEON
    fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, max_threshold_, fast_corners);
#else
    fast::fast_corner_detect_10( image.data, cols, rows, stride, max_threshold_, fast_corners);
#endif

    if(fast_corners.empty())
    {
#if __SSE2__
        fast::fast_corner_detect_10_sse2(img.data, cols, rows, stride, min_threshold_, fast_corners);
#elif HAVE_FAST_NEON
        fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, min_threshold_, fast_corners);
#else
        fast::fast_corner_detect_10(image.data, cols, rows, stride, min_threshold_, fast_corners);
#endif
    }

    std::vector<int> scores, nm_corners;
    fast::fast_corner_score_10(img.data, stride, fast_corners, max_threshold_, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    std::vector<Corner> &corners = corners_in_levels_[level];
    corners.clear();
    corners.reserve(nm_corners.size());

    for(int& index : nm_corners)
    {
        fast::fast_xy& xy = fast_corners[index];
        const int u = xy.x;
        const int v = xy.y;

        //! border check;
        if(u < border_ || v < border_ || u > cols-border_ || v > rows-border_)
            continue;

        //! if the pixel is occupied, ignore this corner
        if(grid_fliter_.getOccupancy(u, v))
            continue;

        const float score = shiTomasiScore(img, u, v);

        //! reject the low-score point
        if(score < eigen_threshold)
            continue;

        const int x = u*scale;
        const int y = v*scale;

        corners.push_back(Corner(x, y, score, level));
    }

    fast_corners.clear();

    return corners.size();
}


void FastDetector::drawGrid(const cv::Mat& img, cv::Mat& img_grid)
{
    img_grid = img.clone();

    int grid_size = grid_fliter_.getSize();
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
float FastDetector::shiTomasiScore(const cv::Mat& img, int u, int v)
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