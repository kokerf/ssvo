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

private:
    int detectGrid(const cv::Mat& image, int level);

    int detectImage(const cv::Mat& image, int level);

    void preSetGrid(const ImgPyr& img_pyr, const std::vector<cv::KeyPoint>& kps);

    void resetGrid();

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

    //! get grid index by pixel in level
    inline int getGridIndex(int u, int v, int level)
    {
        const int scale = (1<<level);
        return (scale*v)/grid_size_*grid_n_cols_ + (scale*u)/grid_size_;
    }

    //! get grid index by pixel in level0
    inline int getGridIndex(int x, int y)
    {
        return y/grid_size_*grid_n_cols_ + x/grid_size_;
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
    const int min_cols_ = 8;
    const int min_rows_ = 8;
    const int max_fts_ = 2;

    int N_;
    int nlevels_;
    int maxThFAST_;
    int minThFAST_;
    int min_score_;

    bool size_ajust_;
    int grid_size_;
    int grid_n_cols_;
    int grid_n_rows_;

    std::vector<cv::Mat> mask_pyr_;
    std::vector<std::vector<cv::KeyPoint> > kps_in_grid_;
};

}//! end of ssvo

#endif