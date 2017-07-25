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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

    Initializer(const cv::Mat& K, const cv::Mat& D);

    InitResult addFirstImage(const cv::Mat& img_ref, std::vector<cv::Point2f>& pts, std::vector<cv::Point2d>& fts);

    InitResult addSecondImage(const cv::Mat& img);

    void getResults(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur,
    std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur, std::vector<Vector3d>& p3ds, cv::Mat& inliers) const;

    void getUndistInilers(std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur) const;

    void getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const;

private:

    void kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur, const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, cv::Mat& inliers);

    void calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Mat& inliers, std::vector<std::pair<int, float> >& disparities);

    bool findBestRT(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t, const Matrix3d& K1, const Matrix3d& K2,
                              const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2, cv::Mat& mask, std::vector<Vector3d>& P3Ds, MatrixXd& T);

    int checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur, const  std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                          const MatrixXd& T, cv::Mat& mask, std::vector<Vector3d>& p3ds, const double sigma2);

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
    static int findFundamentalMat(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d &F,
                                        cv::Mat& inliers, const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

    static inline void computeErrors(const cv::Point2d& p1, const cv::Point2d& p2, Matrix3d& F, double& err1, double& err2);

    static void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, Matrix3d& T);

    static void run8point(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d& F, const bool bE = false);

    static int runRANSAC(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d& F, cv::Mat& inliers,
                         const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

    static void decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t);

};

}


#endif