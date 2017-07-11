#include "feature_detector.hpp"

namespace ssvo{

FastDetector::FastDetector(int N, int maxThFAST, int minThFAST, int min_score, int levels, int grid_size):
    N_(N), maxThFAST_(maxThFAST), minThFAST_(minThFAST), min_score_(min_score), levels_(levels), grid_size_(grid_size)
{
    size_ajust_ = (grid_size_ == -1) ? true : false;
}

void FastDetector::operator()(const ImgPyr& img_pyr, std::vector<cv::KeyPoint>& all_kps, const std::vector<cv::KeyPoint>& ext_kps)
{
    assert(img_pyr.size() == levels_);

    if(size_ajust_)
    {
        grid_size_ = static_cast<int>(std::sqrt(1.0 * img_pyr[0].cols * img_pyr[0].rows / N_));
    }

    grid_n_rows_ = ceil(static_cast<double>(img_pyr[0].rows) / grid_size_);
    grid_n_cols_ = ceil(static_cast<double>(img_pyr[0].cols) / grid_size_);
    if(grid_n_rows_ < min_rows_ || grid_n_cols_ < min_cols_)
    {
        int grid_size0 = img_pyr[0].rows / min_rows_;
        int grid_size1 = img_pyr[0].cols / min_cols_;
        grid_size_ = grid_size0 > grid_size1 ? grid_size0 : grid_size1;

        grid_n_rows_ = ceil(static_cast<double>(img_pyr[0].rows) / grid_size_);
        grid_n_cols_ = ceil(static_cast<double>(img_pyr[0].cols) / grid_size_);
    }

    preSetGrid(img_pyr, ext_kps);

    for(int level = 0; level < levels_; level++)
    {
        // const int rows = img_pyr[level].rows;
        // const int cols = img_pyr[level].cols;

        this->detectImage(img_pyr[level], level);

        // for(cv::KeyPoint &kp : coners)
        // {
        //     const int u = kp.pt.x;
        //     const int v = kp.pt.y;

        //     int n = getGridIndex(u, v);
        // }
    }

    getKeyPointsFromGrid(all_kps);

}

void FastDetector::preSetGrid(const ImgPyr& img_pyr, const std::vector<cv::KeyPoint>& kps)
{
    kps_in_grid_.clear();
    kps_in_grid_.resize(grid_n_rows_*grid_n_cols_);

    for(cv::KeyPoint kp : kps)
    {
        const int n = getGridIndex(kp.pt.x, kp.pt.y);
        const int scale = 1 << kp.octave;
        const int u = round(kp.pt.x / scale);
        const int v = round(kp.pt.y / scale);
        kp.response = shiTomasiScore(img_pyr[kp.octave], u, v);
        insertToGrid(kp, n);
    }
}

int FastDetector::detectGrid(const cv::Mat& image, int level)
{

}

int FastDetector::detectImage(const cv::Mat& image, int level)
{
    assert(image.type() == CV_8UC1);

    const int rows = image.rows;
    const int cols = image.cols;
    const int stride = cols;
    const int scale = 1 << level;

    std::vector<fast::fast_xy> fast_corners;

#if __SSE2__
    fast::fast_corner_detect_10_sse2(image.data, cols, rows, stride, maxThFAST_, fast_corners);
#elif HAVE_FAST_NEON
    fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, maxThFAST_, fast_corners);
#else
    fast::fast_corner_detect_10( image.data, cols, rows, stride, maxThFAST_, fast_corners);
#endif

    if(fast_corners.empty())
    {
#if __SSE2__
        fast::fast_corner_detect_10_sse2(image.data, cols, rows, stride, minThFAST_, fast_corners);
#elif HAVE_FAST_NEON
        fast::fast_corner_detect_9_neon(image.data, cols, rows, stride, minThFAST_, fast_corners);
#else
        fast::fast_corner_detect_10(image.data, cols, rows, stride, minThFAST_, fast_corners);
#endif
    }

    std::vector<int> scores, nm_corners;
    fast::fast_corner_score_10(image.data, cols, fast_corners, maxThFAST_, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(int& index : nm_corners)
    {
        fast::fast_xy& xy = fast_corners[index];
        const int u = xy.x;
        const int v = xy.y;


        const float score = shiTomasiScore(image, u, v);

        if(score < min_score_)
            continue;

        cv::KeyPoint kp(u*scale, v*scale, 0, -1, score, level);
        int n = getGridIndex(kp.pt.x, kp.pt.y);
        insertToGrid(kp, n);
    }

}

void FastDetector::getKeyPointsFromGrid(std::vector<cv::KeyPoint>& all_kps)
{
    all_kps.clear();
    for(std::vector<cv::KeyPoint>& kps : kps_in_grid_)
    {
        const int n = kps.size();
        if (n > max_fts_)
            std::sort(kps.begin(), kps.end(), [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) { return kp1.response > kp2.response; });

        int i = 0;
        for(auto it = kps.begin(), it_end = kps.end(); it != it_end && i < max_fts_; i++)
        {
            all_kps.push_back(*it);
        }
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