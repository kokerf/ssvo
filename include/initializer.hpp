#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include <vector>
#include <opencv2/core.hpp>

#include <Eigen/Dense>

#include "frame.hpp"
#include "config.hpp"

using namespace Eigen;

namespace ssvo{

enum InitResult {RESET=-1, FAILURE=0, SUCCESS=1};

class Initializer
{
public:

    InitResult addFirstFrame(FramePtr frame_ref);

    InitResult addSecondFrame(FramePtr frame_cur);

    void getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur);

private:

    void kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur, std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, std::vector<double>& disparities);

    bool findBestRT(const cv::Mat& R1, const cv::Mat& R2, const cv::Mat& t, const cv::Mat& K1, const cv::Mat& K2,
                    const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3Ds, cv::Mat& T);

    void triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3D);

    void triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2f& pt1, const cv::Point2f& pt2, cv::Mat& P3D);

    void triangulate(const MatrixXf& P1, const MatrixXf& P2, const cv::Point2f& pt1, const cv::Point2f& pt2, Vector4f& P3D);

    //void checkInliers(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& inliers, std::vector<float>& errors);

private:

    FramePtr frame_ref_;
    FramePtr frame_cur_;

    std::vector<cv::Point2f> pts_ref_;
    std::vector<cv::Point2f> pts_cur_;
    std::vector<cv::Point2f> upts_ref_;
    std::vector<cv::Point2f> upts_cur_;
    std::vector<Vector3f> p3ds_;
    std::vector<double> disparities_;
    cv::Mat inliers_;
};

class Fundamental
{
public:
    static int findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat &F,
                                        cv::Mat& inliers, float sigma = 1, int max_iterations = 2000);

    static inline void computeErrors(const cv::Point2f& p1, const cv::Point2f& p2, const float* F, float& err1, float& err2);

    static void Normalize(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_norm, cv::Mat& T);

    static void Normalize(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_norm, Matrix3f& T);

    static void run8point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F);

    static int runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F, cv::Mat& inliers, float sigma = 1, int max_iterations = 2000);

    static void decomposeEssentialMat(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t);

};

}


#endif