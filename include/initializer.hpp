#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include <vector>
#include <opencv2/core.hpp>

#include <Eigen/Dense>

#include "frame.hpp"
#include "config.hpp"
#include "map.hpp"

using namespace Eigen;

namespace ssvo{

enum InitResult {RESET=-1, FAILURE=0, SUCCESS=1};

class Initializer: public noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Initializer> Ptr;

    InitResult addFirstImage(Frame::Ptr frame_ref);

    InitResult addSecondImage(Frame::Ptr frame_cur);

    void reset();

    void createInitalMap(Map::Ptr map, double map_scale=1.0);

    Frame::Ptr getReferenceFrame(){return frame_ref_;}

    void getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const;

    void drowOpticalFlow(const cv::Mat& src, cv::Mat& dst) const;

    static void kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur,
                         const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, cv::Mat& inliers);

    static void calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& inliers, std::vector<std::pair<int, float> >& disparities);

    static bool findBestRT(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t,
                           const Matrix3d& K1, const Matrix3d& K2,
                           const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2,
                           cv::Mat& mask, std::vector<Vector3d>& P3Ds, Matrix<double, 3, 4>& T);

    static int checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur,
                                 const std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                                 const Matrix<double, 3, 4>& T, cv::Mat& mask, std::vector<Vector3d>& p3ds,
                                 const double sigma2);

    static void triangulate(const Matrix<double, 3, 4>& P1, const Matrix<double, 3, 4>& P2,
                            const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D);

    static void reduceVecor(std::vector<cv::Point2f>& pts, const cv::Mat& inliers);

    static void reduceVecor(std::vector<cv::Point2d>& fts, const cv::Mat& inliers);

    inline static Initializer::Ptr create(const FastDetector::Ptr &fast_detector, bool verbose = false)
    {return Initializer::Ptr(new Initializer(fast_detector, verbose));}

private:
    Initializer(const FastDetector::Ptr &fast_detector, bool verbose = false);

private:

    FastDetector::Ptr fast_detector_;
    Frame::Ptr frame_ref_;
    Frame::Ptr frame_cur_;

    Corners corners_;
    std::vector<cv::Point2f> pts_ref_;
    std::vector<cv::Point2f> pts_cur_;
    std::vector<cv::Point2d> fts_ref_;
    std::vector<cv::Point2d> fts_cur_;
    std::vector<Vector3d> p3ds_;
    std::vector<std::pair<int, float> > disparities_;
    cv::Mat inliers_;
    Matrix<double, 3, 4> T_;

    bool finished_;

    bool verbose_;
};

namespace Fundamental
{

int findFundamentalMat(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d &F,
                                    cv::Mat& inliers, const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

inline void computeErrors(const cv::Point2d& p1, const cv::Point2d& p2, Matrix3d& F, double& err1, double& err2);

void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, Matrix3d& T);

void run8point(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d& F, const bool bE = false);

int runRANSAC(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next, Matrix3d& F, cv::Mat& inliers,
                     const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

void decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t);

}//! namespace Fundamental

}//! namspace ssvo

#endif