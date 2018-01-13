#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "config.hpp"

using namespace Eigen;

namespace ssvo{

//! fifo size = 4
//!            /last
//!   | [] [] [] [] |
//! ref/         \cur
class FrameCandidate{
public:
    typedef std::shared_ptr<FrameCandidate> Ptr;
    Frame::Ptr frame;
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2d> fts;
    std::vector<int> level;
    std::vector<int64_t> idx;
    static const int size;

    void createFts();
    int getInliers(std::vector<bool> &inliers);
    int updateInliers(const std::vector<bool> &inliers);
    int checkTracking(const int min_idx, const int max_idx, const int min_track);
    void getMatch(std::vector<bool> &mask, const int ref_idx);

    inline static Ptr create(const Frame::Ptr &frame)
    {return std::make_shared<FrameCandidate>(FrameCandidate(frame));}
    inline static Ptr create(const Frame::Ptr &frame, const FrameCandidate::Ptr &cand)
    {return std::make_shared<FrameCandidate>(FrameCandidate(frame, cand));}

private:
    FrameCandidate(const Frame::Ptr &frame);
    FrameCandidate(const Frame::Ptr &frame, const FrameCandidate::Ptr &cand);
};

class Initializer: public noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum Result {RESET=-2, READY=-1, FAILURE=0, SUCCESS=1};

    typedef std::shared_ptr<Initializer> Ptr;

    Result addFirstImage(Frame::Ptr frame_ref);

    Result addImage(Frame::Ptr frame_cur);

    void reset();

    void createInitalMap(double map_scale=1.0);

    Frame::Ptr getReferenceFrame(){return cand_ref_->frame;}

    void getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const;

    void drowOpticalFlow(cv::Mat& dst) const;

    void drowOpticalFlowMatch(cv::Mat& dst) const;

    static void calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                              const std::vector<bool> &mask, std::vector<std::pair<int, float> >& disparities);

    static bool findBestRT(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t,
                           const Matrix3d& K1, const Matrix3d& K2,
                           const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2,
                           std::vector<bool>& mask, std::vector<Vector3d>& P3Ds, Matrix<double, 3, 4>& T);

    static int checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur,
                                 const std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                                 const Matrix<double, 3, 4>& T, std::vector<bool>& mask, std::vector<Vector3d>& p3ds,
                                 const double sigma2);

    static void triangulate(const Matrix<double, 3, 4>& P1, const Matrix<double, 3, 4>& P2,
                            const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D);

    inline static Initializer::Ptr create(const FastDetector::Ptr &fast_detector, bool verbose = false)
    {return Initializer::Ptr(new Initializer(fast_detector, verbose));}

private:

    Initializer(const FastDetector::Ptr &fast_detector, bool verbose = false);

    Result createNewCorners(const FrameCandidate::Ptr &candidate);

    bool changeReference(int buffer_offset);

private:

    FastDetector::Ptr fast_detector_;

    std::deque<FrameCandidate::Ptr> frame_buffer_;

    FrameCandidate::Ptr cand_ref_;//！ front of frame_buffer_
    FrameCandidate::Ptr cand_cur_;
    FrameCandidate::Ptr cand_last_;
    std::vector<Vector3d> p3ds_;
    std::vector<std::pair<int, float> > disparities_;
    std::vector<bool> inliers_;
    Matrix<double, 3, 4> T_;

    bool finished_;

    bool verbose_;
};

}//! namspace ssvo

#endif