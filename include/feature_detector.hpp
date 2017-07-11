#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "fast.h"



typedef std::vector<cv::Mat> ImgPyr;

namespace ssvo
{

class FastDetector
{
public:
    FastDetector(int N, int maxThFAST = 20, int minThFAST = 5, int min_score = 20, int levels = 3, int grid_size = -1);

    void operator()(const ImgPyr& img_pyr, std::vector<cv::KeyPoint>& all_kps, const std::vector<cv::KeyPoint>& ext_kps);

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
            for(auto it=grid.begin(); it!=grid.end(); it++)
            {
                if(it->response < kp.response)
                {
                    grid.insert(it, kp);
                    if(grid.size() > max_fts_)
                        grid.resize(max_fts_);
                    return true;
                }
            }
        }

        return false;
    }

    inline int getGridIndex(int x, int y, int level)
    {
        const int scale = (1<<level);
        return (scale*y)/grid_size_*grid_n_cols_ + (scale*x)/grid_size_;
    }

    inline int getGridIndex(int x, int y)
    {
        return y/grid_size_*grid_n_cols_ + x/grid_size_;
    }

private:
    const int border_ = 8;
    const int min_cols_ = 8;
    const int min_rows_ = 8;
    const int max_fts_ = 1;

    int N_;
    int maxThFAST_;
    int minThFAST_;
    int min_score_;
    int levels_;

    bool size_ajust_;
    int grid_size_;
    int grid_n_cols_;
    int grid_n_rows_;

    std::vector<std::vector<cv::KeyPoint> > kps_in_grid_;
};

}//! end of ssvo

#endif