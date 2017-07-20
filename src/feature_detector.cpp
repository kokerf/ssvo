#include <iostream>
#include <opencv2/imgproc.hpp>

#include "feature_detector.hpp"

namespace ssvo{

FastDetector::FastDetector(int N, int top_level, bool size_ajust):
    N_(N), nlevels_(top_level+1), size_ajust_(size_ajust)
{
    grid_size_ = -1;
}

int FastDetector::detectByImage(const ImgPyr& img_pyr, std::vector<cv::KeyPoint>& all_kps, const std::vector<cv::KeyPoint>& ext_kps)
{
    preProccess(img_pyr, ext_kps);

    new_coners_ = 0;
    for(int level = 0; level < nlevels_; level++)
    {
        new_coners_ += this->detectByImage(img_pyr[level], level);
    }

    getKeyPointsFromGrid(img_pyr[0], all_kps);

    if(size_ajust_)
    {
        int size = grid_size_;
        if(all_kps.size() < N_*0.95)
            size -= 5;
        else if(all_kps.size() > N_*1.1)
            size += 5;

        resetGridSize(size);
    }

    return new_coners_;
}

int FastDetector::detectByGrid(const ImgPyr& img_pyr, std::vector<cv::KeyPoint>& all_kps, const std::vector<cv::KeyPoint>& ext_kps)
{
    preProccess(img_pyr, ext_kps);

    new_coners_ = 0;
    for(int level = 0; level < nlevels_; level++)
    {
        new_coners_ += this->detectByGrid(img_pyr[level], level);
    }

    getKeyPointsFromGrid(img_pyr[0], all_kps);

    if(size_ajust_)
    {
        int size = grid_size_;
        if(all_kps.size() < N_*0.95)
            size -= 5;
        else if(all_kps.size() > N_*1.1)
            size += 5;

        resetGridSize(size);
    }

    return new_coners_;
}

void FastDetector::preProccess(const ImgPyr& img_pyr, const std::vector<cv::KeyPoint>& kps)
{
    width_ = img_pyr[0].cols;
    height_ = img_pyr[0].rows;

    //! get max level
    nlevels_ = MIN(img_pyr.size(), nlevels_);

    //! creat grid coordinate
    creatGrid();

    //! set mask and grid
    kps_in_grid_.clear();
    kps_in_grid_.resize(grid_n_rows_*grid_n_cols_);

    occupancy_grid_.resize(grid_n_rows_*grid_n_cols_, 0);

    for(const cv::Mat& image : img_pyr)
    {
        mask_pyr_.push_back(cv::Mat::zeros(image.size(), CV_8UC1));
    }

    for(cv::KeyPoint kp : kps)
    {
        const int n = getGridIndex(kp.pt.x, kp.pt.y);
        const int level = kp.octave;
        const int scale = 1 << level;
        const int u = round(kp.pt.x / scale);
        const int v = round(kp.pt.y / scale);
        //! compute score and insert to grid
        kp.response = shiTomasiScore(img_pyr[level], u, v);
        insertToGrid(kp, n);
        //! set image mask
        setMask(u, v, level);
        occupancy_grid_[n] ++;
    }
}

void FastDetector::resetGridSize(const int size)
{
    grid_size_ = MAX(size, Config::gridMinSize());
    grid_n_rows_ = floorf(static_cast<double>(height_) / grid_size_);
    grid_n_cols_ = floorf(static_cast<double>(width_) / grid_size_);

    offset_rows_ = (height_ - grid_size_ * grid_n_rows_) >> 1;
    offset_cols_ = (width_ - grid_size_ * grid_n_cols_) >> 1;
}

void FastDetector::creatGrid()
{
    //! adjust grid size
    if(grid_size_ == -1)
    {
        const int size = floorf(std::sqrt(1.0 * width_ * height_ / (N_/Config::gridMaxFeatures())));
        resetGridSize(size);
    }

    grid_boundary_x_.resize(grid_n_cols_-1);
    grid_boundary_y_.resize(grid_n_rows_-1);
    for(int i = 0; i < grid_n_cols_-1; ++i)
    {
        const int x = offset_cols_ + (i+1) * grid_size_;
        grid_boundary_x_[i] = x;
    }

    for(int i = 0; i < grid_n_rows_-1; ++i)
    {
        const int y = offset_rows_ + (i+1) * grid_size_;
        grid_boundary_y_[i] = y;
    }
}

int FastDetector::detectByGrid(const cv::Mat& img, int level)
{
    assert(img.type() == CV_8UC1);

    const int scale = 1 << level;
    const int grid_size =  floorf(static_cast<double>(grid_size_) / scale);
    if(grid_size < Config::gridMinSize())
        return this->detectByImage(img, level);

    const int rows = img.rows;
    const int cols = img.cols;
    const int stride = cols;

    const int n_grids = grid_n_rows_*grid_n_cols_;
    int new_coners = 0;
    for(int i = 0; i < n_grids; ++i)
    {
        if(occupancy_grid_[i] >= Config::gridMaxFeatures())
            continue;

        const int i_rows = i / grid_n_cols_;
        const int i_cols = i % grid_n_cols_;

        const bool b_left = (i_cols == 0);
        const bool b_right = (i_cols == grid_n_cols_-1);
        const bool b_top = (i_rows == 0);
        const bool b_bottom = (i_rows == grid_n_rows_-1);

        const int x_start = b_left ? 0 : (grid_boundary_x_[i_cols-1] >> level) - 3;
        const int x_end = b_right ? cols : (grid_boundary_x_[i_cols] >> level) + 3;
        const int y_start = b_top ? 0 : (grid_boundary_y_[i_rows-1] >> level) - 3;
        const int y_end = b_bottom ? rows : (grid_boundary_y_[i_rows] >> level) + 3;

        const int grid_cols = x_end - x_start;
        const int grid_rows = y_end - y_start;

        const uchar* img_ptr = &img.ptr<uchar>(y_start)[x_start];

        std::vector<fast::fast_xy> fast_corners;

#if __SSE2__
        fast::fast_corner_detect_10_sse2(img_ptr, grid_cols, grid_rows, stride, Config::fastMaxThreshold(), fast_corners);
#elif HAVE_FAST_NEON
        fast::fast_corner_detect_9_neon(img_ptr, grid_cols, grid_rows, stride, maxThFAST_, fast_corners);
#else
        fast::fast_corner_detect_10( img_ptr, grid_cols, grid_rows, stride, maxThFAST_, fast_corners);
#endif

        if(fast_corners.empty())
        {
#if __SSE2__
            fast::fast_corner_detect_10_sse2(img_ptr, grid_cols, grid_rows, stride, Config::fastMinThreshold(), fast_corners);
#elif HAVE_FAST_NEON
            fast::fast_corner_detect_9_neon(img_ptr, grid_cols, grid_rows, stride, minThFAST_, fast_corners);
#else
            fast::fast_corner_detect_10(img_ptr, grid_cols, grid_rows, stride, minThFAST_, fast_corners);
#endif
        }

        std::vector<int> scores, nm_corners;
        fast::fast_corner_score_10(img_ptr, stride, fast_corners, Config::fastMaxThreshold(), scores);
        fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

        for(int& index : nm_corners)
        {
            fast::fast_xy& xy = fast_corners[index];
            const int u = xy.x + x_start;
            const int v = xy.y + y_start;

            //! border check;
            if(u < Config::imageBorder() || v < Config::imageBorder() || u > cols-Config::imageBorder() || v > rows-Config::imageBorder())
                continue;

            //! if the pixel is occupied, ignore this corner
            if(mask_pyr_[level].ptr<uchar>(v)[u])
                continue;

            const float score = shiTomasiScore(img, u, v);

            //! reject the low-score point
            if(score < Config::fastMinEigen())
                continue;

            new_coners ++;
            const int x = u*scale;
            const int y = v*scale;
            cv::KeyPoint kp(x, y, 0, -1, score, level);
            //! some point will be in the boundary
            const int n = getGridIndex(x, y);
            insertToGrid(kp, n);
        }

    }

    return new_coners;
}

int FastDetector::detectByImage(const cv::Mat& img, int level)
{
    assert(img.type() == CV_8UC1);

    const int rows = img.rows;
    const int cols = img.cols;
    const int stride = cols;
    const int scale = 1 << level;

    std::vector<fast::fast_xy> fast_corners;

#if __SSE2__
    fast::fast_corner_detect_10_sse2(img.data, cols, rows, stride, Config::fastMaxThreshold(), fast_corners);
#elif HAVE_FAST_NEON
    fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, maxThFAST_, fast_corners);
#else
    fast::fast_corner_detect_10( image.data, cols, rows, stride, maxThFAST_, fast_corners);
#endif

    if(fast_corners.empty())
    {
#if __SSE2__
        fast::fast_corner_detect_10_sse2(img.data, cols, rows, stride, Config::fastMinThreshold(), fast_corners);
#elif HAVE_FAST_NEON
        fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, minThFAST_, fast_corners);
#else
        fast::fast_corner_detect_10(image.data, cols, rows, stride, minThFAST_, fast_corners);
#endif
    }

    std::vector<int> scores, nm_corners;
    fast::fast_corner_score_10(img.data, stride, fast_corners, Config::fastMaxThreshold(), scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    int new_coners = 0;
    for(int& index : nm_corners)
    {
        fast::fast_xy& xy = fast_corners[index];
        const int u = xy.x;
        const int v = xy.y;

        //! border check;
        if(u < Config::imageBorder() || v < Config::imageBorder() || u > cols-Config::imageBorder() || v > rows-Config::imageBorder())
            continue;

        //! if the pixel is occupied, ignore this corner
        if(mask_pyr_[level].ptr<uchar>(v)[u])
            continue;

        const float score = shiTomasiScore(img, u, v);

        //! reject the low-score point
        if(score < Config::fastMinEigen())
            continue;

        new_coners ++;
        const int x = u*scale;
        const int y = v*scale;
        cv::KeyPoint kp(x, y, 0, -1, score, level);
        const int n = getGridIndex(x, y);
        insertToGrid(kp, n);
    }

    return new_coners;
}

void FastDetector::getKeyPointsFromGrid(const cv::Mat& img, std::vector<cv::KeyPoint>& all_kps)
{
    all_kps.clear();
    for(std::vector<cv::KeyPoint>& kps : kps_in_grid_)
    {
        if(kps.empty())
            continue;

        const int size = kps.size();
        int nfts= 0;
        for(int n = 0; n < size && nfts < Config::gridMaxFeatures(); n++)
        {
            all_kps.push_back(kps[n]);
            nfts++;
        }
     }
}

void FastDetector::drawGrid(const cv::Mat& img, cv::Mat& img_grid)
{
    img_grid = img.clone();

    for(int c = 1; c < grid_n_cols_; c++)
    {
        cv::Point start(offset_cols_ + grid_size_*c-1, 0);
        cv::Point end(offset_cols_ + grid_size_*c-1, img.rows-1);

        cv::line(img_grid, start, end, cv::Scalar(0, 50, 0));
    }

    for(int l = 1; l < grid_n_rows_; l++)
    {
        cv::Point start(0, offset_rows_ + grid_size_*l-1);
        cv::Point end(img.cols-1, offset_rows_ + grid_size_*l-1);

        cv::line(img_grid, start, end, cv::Scalar(0, 50, 0));
    }
}

//! site from rpg_vikit
//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/src/vision.cpp#L113
//! opencv ref: https://github.com/opencv/opencv/blob/26be2402a3ad6c9eacf7ba7ab2dfb111206fffbb/modules/imgproc/src/corner.cpp#L129
float FastDetector::shiTomasiScore(const cv::Mat& img, int u, int v)
{
    assert(img.type() == CV_8UC1);

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