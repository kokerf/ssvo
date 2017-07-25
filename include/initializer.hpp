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

    Initializer(const cv::Mat& K, const cv::Mat& D);

    InitResult addFirstImage(const cv::Mat& img_ref, std::vector<cv::Point2f>& pts, std::vector<cv::Point2d>& fts);

    InitResult addSecondImage(const cv::Mat& img);

    void getResults(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur,
    std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur, std::vector<Vector3d>& p3ds) const;

    void getUndistInilers(std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur) const;

    void getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const;

private:

    void kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur, const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, cv::Mat& inliers);

    void calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Mat& inliers, std::vector<std::pair<int, float> >& disparities);

    bool findBestRT(const cv::Mat& R1, const cv::Mat& R2, const cv::Mat& t, const cv::Mat& K1, const cv::Mat& K2,
                    const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2, cv::Mat& mask, cv::Mat& P3Ds, cv::Mat& T);

    int checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur, const std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                           const cv::Mat& T, cv::Mat& mask, const cv::Mat& P3Ds, const double sigma2, std::vector<Vector3d>& vP3Ds);

    void triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2, cv::Mat& mask, cv::Mat& P3D);

    void triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2d& ft1, const cv::Point2d& ft2, cv::Mat& P3D);

    void triangulate(const MatrixXd& P1, const MatrixXd& P2, const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D);

    void reduceVecor(std::vector<cv::Point2f>& pts, const cv::Mat& inliers);

    void reduceVecor(std::vector<cv::Point2d>& fts, const cv::Mat& inliers);

private:

    cv::Mat K_, D_;
    cv::Mat img_ref_;
    cv::Mat img_cur_;

    std::vector<cv::Point2f> pts_ref_;
    std::vector<cv::Point2f> pts_cur_;
    std::vector<cv::Point2d> fts_ref_;
    std::vector<cv::Point2d> fts_cur_;
    std::vector<Vector3d> p3ds_;
    std::vector<std::pair<int, float> > disparities_;
    cv::Mat inliers_;

    bool finished_;
};

class Fundamental
{
public:
    static int findFundamentalMat(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, cv::Mat &F,
                                        cv::Mat& inliers, const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

    static inline void computeErrors(const cv::Point2d& p1, const cv::Point2d& p2, const double* F, double& err1, double& err2);

    static void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, cv::Mat& T);

    static void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, Matrix3f& T);

    static void run8point(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, cv::Mat& F, const bool bE = false);

    static int runRANSAC(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, cv::Mat& F, cv::Mat& inliers,
                         const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

    static void decomposeEssentialMat(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t);

};

}


#endif