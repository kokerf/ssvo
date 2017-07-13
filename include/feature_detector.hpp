#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "fast.h"



typedef std::vector<cv::Mat> ImgPyr;

//! (u,v) is in the n-level image of pyramid
//! (x,y) is in the 0-level image of pyramid
namespace ssvo
{

class FastDetector
{
public:
    FastDetector(int N, int top_level = 3, int maxThFAST = 20, int minThFAST = 5, int min_score = 20, int grid_size = -1);

    void operator()(const ImgPyr& img_pyr, std::vector<cv::KeyPoint>& all_kps, const std::vector<cv::KeyPoint>& ext_kps);

    void drawGrid(const cv::Mat& img, cv::Mat& img_grid);

private:
    int detectByGrid(const cv::Mat& image, int level);

    int detectByImage(const cv::Mat& image, int level);

    void preProccess(const ImgPyr& img_pyr, const std::vector<cv::KeyPoint>& kps);

    void creatGrid();

    void getKeyPointsFromGrid(std::vector<cv::KeyPoint>& all_kps);

    float shiTomasiScore(const cv::Mat& img, int u, int v);

    inline bool insertToGrid(cv::KeyPoint& kp, int n)
    {
        std::vector<cv::KeyPoint> &grid = kps_in_grid_[n];

        if(grid.empty())
        {
            grid.push_back(kp);
            return true;
        }
        else
        {
            std::vector<cv::KeyPoint>::iterator it;
            for(it = grid.begin(); it!=grid.end(); it++)
            {
                if(it->response < kp.response)
                    break;
            }

            grid.insert(it, kp);

            return true;
        }

        return false;
    }

    //! get grid index by pixel in level0
    inline int getGridIndex(int x, int y)
    {
        y = y > offset_rows_ ? y-offset_rows_ : y;
        x = x > offset_cols_ ? x-offset_cols_ : 0;
        int n_rows = y/grid_size_;
        int n_cols = x/grid_size_;
        n_rows = n_rows >= grid_n_rows_ ? grid_n_rows_-1: n_rows;
        n_cols = n_cols >= grid_n_cols_ ? grid_n_cols_-1: n_cols;

        return n_rows*grid_n_cols_ + n_cols;
    }

    //! get grid index by pixel in level
    inline int getGridIndex(int u, int v, int level)
    {
        const int scale = (1<<level);
        return getGridIndex(u*scale, v*scale);
    }

    //! set 3×3 mask in mask image
    inline bool setMask(int u, int v, int level)
    {
        //! Attention!!! no boundary check!
        mask_pyr_[level].ptr<uchar>(v-1)[u-1] = 255;
        mask_pyr_[level].ptr<uchar>(v-1)[u] = 255;
        mask_pyr_[level].ptr<uchar>(v-1)[u+1] = 255;
        mask_pyr_[level].ptr<uchar>(v)[u-1] = 255;
        mask_pyr_[level].ptr<uchar>(v)[u] = 255;
        mask_pyr_[level].ptr<uchar>(v)[u+1] = 255;
        mask_pyr_[level].ptr<uchar>(v+1)[u-1] = 255;
        mask_pyr_[level].ptr<uchar>(v+1)[u] = 255;
        mask_pyr_[level].ptr<uchar>(v+1)[u+1] = 255;
        return true;
    }

public:
    int new_coners_;

private:
    const int border_ = 4;
    const int min_size_ = 8;
    const int max_fts_ = 1;

    int N_;
    int nlevels_;
    int maxThFAST_;
    int minThFAST_;
    int min_score_;

    bool size_ajust_;
    int grid_size_;
    int grid_n_cols_;
    int grid_n_rows_;
    int offset_cols_;
    int offset_rows_;

    std::vector<cv::Mat> mask_pyr_;
    std::vector<int> occupancy_grid_;
    std::vector<std::vector<cv::KeyPoint> > kps_in_grid_;
};

}//! end of ssvo

#endif