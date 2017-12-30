#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
#include "global.hpp"

#include "initializer.hpp"
#include <opencv/cxeigen.hpp>

namespace ssvo{

const int FrameCandidate::size = 200;

FrameCandidate::FrameCandidate(const Frame::Ptr &frame) :
    frame(frame)
{
    corners.resize(size);
    pts.resize(size, cv::Point2f(0, 0));
    idx.resize(size, -1);
}

FrameCandidate::FrameCandidate(const Frame::Ptr &frame, const FrameCandidate::Ptr &cand) :
    frame(frame), pts(cand->pts), corners(cand->corners), idx(cand->idx)
{}

void FrameCandidate::update(const Frame::Ptr &frame)
{
    this->frame = frame;
}

void FrameCandidate::createFts()
{
    std::vector<cv::Point2f> temp_udist;
    cv::undistortPoints(pts, temp_udist, frame->cam_->cvK(), frame->cam_->cvD());
    fts.resize(size);
    for(int i = 0; i < FrameCandidate::size; ++i)
    {
        fts[i].x = (double)temp_udist[i].x;
        fts[i].y = (double)temp_udist[i].y;
    }
}

int FrameCandidate::updateInliers(const std::vector<bool> &inliers)
{
    int count = 0;
    for(int i = 0; i < size; ++i)
    {
        if(inliers[i])
        {
            count++;
            continue;
        }

        idx[i] = -1;
        pts[i].x = 0;
        pts[i].y = 0;
        corners[i] = Corner(0,0,0,0);
    }
    return count;
}

void FrameCandidate::getMatch(std::vector<bool> &mask, const int ref_idx)
{
    mask.resize(size);
    for(int i = 0; i < size; ++i)
    {
        if(idx[i] != -1 && idx[i] <= ref_idx)
            mask[i] = true;
        else
            mask[i] = false;
    }
}

int FrameCandidate::checkReference(const int min_idx, const int max_idx, const int min_track)
{
    assert(min_idx <= max_idx);

    const int N = max_idx-min_idx;
    std::vector<int> count(N, 0);
    for(int i = 0; i < size; ++i)
    {
        if(idx[i] == -1)
            continue;

        if(idx[i] < min_idx)
            idx[i] = min_idx;

        count[idx[i]-min_idx]++;
    }


    int delta = max_idx-1;
    int corners_count = 0;
    for(int i = 0; i < N; ++i)
    {
        corners_count += count[i];
        if(corners_count < min_track)
            continue;

        delta = i;
        break;
    }

    return delta;
}

Initializer::Initializer(const FastDetector::Ptr &fast_detector, bool verbose):
    fast_detector_(fast_detector), cand_ref_(nullptr), cand_cur_(nullptr), cand_last_(nullptr), verbose_(verbose)
{};


void Initializer::reset()
{
//    pts_ref_.clear();
//    fts_ref_.clear();
//    pts_cur_.clear();
//    fts_cur_.clear();
//    p3ds_.clear();
//    disparities_.clear();
//    inliers_.release();
//    finished_ = false;
//
//    frame_ref_.reset();
}

InitResult Initializer::createNewCorners(const FrameCandidate::Ptr &candidate)
{
    Corners old_corners;
    for(int i = 0; i < FrameCandidate::size; ++i)
    {
        if(candidate->idx[i] < 0)
            continue;

        old_corners.emplace_back(Corner(candidate->pts[i].x, candidate->pts[i].y, 0, 0));
    }

    Corners new_corners;
    fast_detector_->detect(candidate->frame->images(), new_corners, old_corners, FrameCandidate::size/0.85, Config::fastMinEigen());

    //! check corner number of first image
    if(old_corners.size() + new_corners.size() < FrameCandidate::size)
    {
        return RESET;
    }

    const int needed = FrameCandidate::size - old_corners.size();
    std::nth_element(new_corners.begin(), new_corners.begin()+needed, new_corners.end(),
                     [](const Corner &c1, const Corner &c2){ return c1.score > c2.score; });

    new_corners.resize(needed);
    auto new_corners_itr = new_corners.begin();
    for(int i = 0; i < FrameCandidate::size; ++i, ++new_corners_itr)
    {
        if(candidate->idx[i] >= 0)
            continue;

        const Corner &corner = *new_corners_itr;
        candidate->idx[i] = candidate->frame->id_;
        candidate->corners[i] = *new_corners_itr;
        candidate->pts[i].x = corner.x;
        candidate->pts[i].y = corner.y;
    }

    LOG_IF(INFO, verbose_) << "[INIT][0] New Detect corners: " << new_corners.size();

    return SUCCESS;
}

bool Initializer::changeReference(int buffer_offset)
{
    if(buffer_offset <= 0)
        return false;

    LOG_IF(INFO, verbose_) << "[INIT][1] Change reference frame with buffer offset: " << buffer_offset << " / " << frame_buffer_.size();
    for(int i = buffer_offset; i > 0; i--)
    {
        frame_buffer_.pop_front();
    }

    if(frame_buffer_.empty())
    {
        frame_buffer_.push_back(cand_cur_);
        cand_ref_ = cand_cur_;
        cand_ref_->createFts();
        return true;
    }

    cand_ref_ = frame_buffer_.front();
    cand_ref_->createFts();

    //! change reference index in current
    const int64_t ref_index = cand_ref_->frame->id_;
    std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < FrameCandidate::size; i++)
    {
        if(index[i] != -1 && index[i] < ref_index)
            index[i] = ref_index;
    }

    return false;
}

InitResult Initializer::addImage(Frame::Ptr frame_cur)
{
//    if(finished_)
//    {
//        LOG(ERROR) << "[INIT][*] Last initialization is succeed! Plesase reset!";
//        return RESET;
//    }

    LOG_IF(INFO, verbose_) << "[INIT][*] Processing Frame: " << frame_cur->id_;


    //! create first candidate
    if(frame_buffer_.empty())
    {
        frame_buffer_.push_back(FrameCandidate::create(frame_cur));
        cand_ref_ = frame_buffer_.front();
        createNewCorners(cand_ref_);
        cand_ref_->createFts();
        cand_cur_ = frame_buffer_.back();

        return SUCCESS;
    }

    //! rest current candidate
    cand_last_ = frame_buffer_.back();
    cand_cur_ = FrameCandidate::create(frame_cur, cand_last_);
    LOG_IF(INFO, verbose_) << "[INIT][*] Ref: " << cand_ref_->frame->id_
                           << ", Lst: " << cand_last_->frame->id_
                           << ", Cur: " << cand_cur_->frame->id_;

    double t1 = (double)cv::getTickCount();

    //! [1] KLT tracking
    kltTrack(cand_last_->frame->opticalImages(), cand_cur_->frame->opticalImages(), Frame::optical_win_size_, cand_last_->pts, cand_cur_->pts, inliers_, true);
    cand_cur_->updateInliers(inliers_);
    //! if track too little corners in reference, then change reference
    const int offset = cand_cur_->checkReference(cand_ref_->frame->id_, cand_cur_->frame->id_, Config::initMinTracked());
    //! return if the buffer is empty
    if(changeReference(offset))
        return RESET;

    std::vector<cv::Point2f> ref_pts;
    std::vector<cv::Point2f> cur_pts;
    getTrackedPoints(ref_pts, cur_pts);
    cand_cur_->getMatch(inliers_, cand_ref_->frame->id_);
    calcDisparity(ref_pts, cur_pts, inliers_, disparities_);
//    LOG_IF(INFO, verbose_) << "[INIT][1] KLT tracking points: " << disparities_.size();
//    if(disparities_.size() < Config::initMinTracked()) return RESET;

    double t2 = (double)cv::getTickCount();

    //! [2] calculate disparities, remove outliers
    std::vector<std::pair<int, float> > disparities_temp = disparities_;
    std::sort(disparities_temp.begin(), disparities_temp.end(), [](const std::pair<int, float> disp1, const std::pair<int, float> disp2){return (disp1.second < disp2.second);});
    float disparity = disparities_temp.at(disparities_temp.size()/2).second;
    float max_disparity = disparities_temp.at(disparities_temp.size()*3/4).second * 2;
    for(size_t i = 0; i < disparities_.size(); ++i)
    {
        if(disparities_[i].second > max_disparity)
        {
            const int id = disparities_[i].first;
            inliers_[id] = false;
            cand_cur_->idx[id] = -1;
        }
    }
    LOG_IF(INFO, verbose_) << "[INIT][2] Avage disparity: " << disparity;
    if(disparity < Config::initMinDisparity()) return FAILURE;

    double t3 = (double)cv::getTickCount();

    //! [3] geometry check by F matrix
    //! find fundamental matrix
    Matrix3d E;
    cand_cur_->createFts(); //! get undistorted points
    int inliers_count = Fundamental::findFundamentalMat(cand_ref_->fts, cand_cur_->fts, E, inliers_, Config::pixelUnSigma2(), Config::initMaxRansacIters(), true);

    LOG_IF(INFO, verbose_) << "[INIT][3] Inliers after epipolar geometry check: " << inliers_count;
    if(inliers_count < Config::initMinInliers()) return FAILURE;
//
//    double t4 = (double)cv::getTickCount();
//
//    //! [4] cheirality check
//    Matrix3d R1, R2;
//    Vector3d t;
//    Fundamental::decomposeEssentialMat(E, R1, R2, t);
//
//    Matrix3d K = Matrix3d::Identity(3,3);
//    bool succeed = findBestRT(R1, R2, t, K, K, fts_ref_, fts_cur_, inliers_, p3ds_, T_);
//    if(!succeed) return FAILURE;
//    LOG_IF(INFO, verbose_) << "[INIT][4] Inliers after cheirality check: " << cv::countNonZero(inliers_);
//
//    double t5 = (double)cv::getTickCount();
//
//    //! [5] reprojective check
//    inliers_count = checkReprejectErr(pts_ref_, pts_cur_, fts_ref_, fts_cur_, T_, inliers_, p3ds_, Config::pixelUnSigma2()*4);
//    LOG_IF(INFO, verbose_) << "[INIT][5] Inliers after reprojective check: " << inliers_count;
//    if(inliers_count < Config::initMinInliers()) return FAILURE;
//
//    double t6 = (double)cv::getTickCount();
//    LOG_IF(WARNING, verbose_) << "[INIT][*] Time: " << (t2-t1)/cv::getTickFrequency() << " "
//                              << (t3-t2)/cv::getTickFrequency() << " "
//                              << (t4-t3)/cv::getTickFrequency() << " "
//                              << (t5-t4)/cv::getTickFrequency() << " "
//                              << (t6-t5)/cv::getTickFrequency();
//
//    finished_ = true;

    int inlier_num = std::count(cand_cur_->idx.begin(), cand_cur_->idx.end(), -1);
    if(inlier_num > 0)
        createNewCorners(cand_cur_);
    frame_buffer_.push_back(cand_cur_);
    return FAILURE;
}

void Initializer::createInitalMap(std::vector<Vector3d> &points, double map_scale)
{
    //! [6] create inital map
    const size_t N = cand_ref_->pts.size();

    //! calculate mean depth
    double mean_depth = 0.0;
    size_t count = 0;
    for(size_t i = 0; i < N; ++i)
    {
        if(inliers_[i])
        {
            mean_depth += p3ds_[i][2];
            count++;
        }
    }
    double scale = map_scale*count/mean_depth;

    //! rescale frame pose
    SE3d T_cur_from_ref(T_.topLeftCorner(3,3), T_.rightCols(1));
    SE3d T_ref_from_cur = T_cur_from_ref.inverse();
    T_ref_from_cur.translation() = T_ref_from_cur.translation() * scale;
    cand_ref_->frame->setPose(Matrix3d::Identity(), Vector3d::Zero());
    cand_cur_->frame->setPose(T_ref_from_cur);

    //! create and rescale map points
    const std::vector<int64_t> &index = cand_cur_->idx;

//    points.reserve(count);
//    for(size_t i = 0; i < N; ++i)
//    {
//        if(!inliers_[i])
//            continue;
//
//        Vector2d px_ref(pts_ref_[i].x, pts_ref_[i].y);
//        Vector2d px_cur(pts_cur_[i].x, pts_cur_[i].y);
//        Vector3d ft_ref(fts_ref_[i].x, fts_ref_[i].y, 1);
//        Vector3d ft_cur(fts_cur_[i].x, fts_cur_[i].y, 1);
//
//        points.emplace_back(p3ds_[i]*scale);
//
//        Feature::Ptr feature_ref = Feature::create(px_ref, ft_ref.normalized(), corners_[i].level, nullptr);
//        Feature::Ptr feature_cur = Feature::create(px_cur, ft_cur.normalized(), corners_[i].level, nullptr);
//
//        frame_ref_->addFeature(feature_ref);
//        frame_cur_->addFeature(feature_cur);
//    }
}

void Initializer::getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const
{
    const int ref_id = cand_ref_->frame->id_;
    const size_t track_points = std::count(cand_cur_->idx.begin(), cand_cur_->idx.end(), ref_id);

    pts_ref.clear();
    pts_cur.clear();
    pts_ref.reserve(track_points);
    pts_cur.reserve(track_points);

    const std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < FrameCandidate::size; ++i)
    {
        if(index[i] < 0 || index[i] > ref_id)
            continue;

        pts_ref.push_back(cand_ref_->pts[i]);
        pts_cur.push_back(cand_cur_->pts[i]);
    }
}

void Initializer::drowOpticalFlow(const cv::Mat &src, cv::Mat &dst) const
{
    std::vector<cv::Point2f> pts_ref, pts_cur;
    getTrackedPoints(pts_ref, pts_cur);

    dst = src.clone();
    for(size_t i=0; i<pts_ref.size();i++)
    {
        cv::line(dst, pts_ref[i], pts_cur[i], cv::Scalar(0, 255, 0));
    }
}

void Initializer::drowOpticalFlowMatch(cv::Mat &dst) const
{
    const cv::Mat &cur = cand_cur_->frame->getImage(0);
    const cv::Mat &ref = cand_ref_->frame->getImage(0);
    const int cols = cur.cols;
    const int rows = cur.rows;
    dst = cv::Mat(rows, cols*2, CV_8UC3);
    cv::cvtColor(ref, dst.colRange(0, cols), CV_GRAY2RGB);
    cv::cvtColor(cur, dst.colRange(cols, 2*cols), CV_GRAY2RGB);

    const int N = FrameCandidate::size;
    const int ref_id = cand_ref_->frame->id_;
    const std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < N; ++i)
    {
        if(index[i] < 0 || index[i] > ref_id)
            continue;

        cv::RNG rng(i);
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::line(dst, cand_ref_->pts[i], cand_cur_->pts[i]+cv::Point2f(cols, 0), color);
    }
}

void Initializer::kltTrack(const ImgPyr &imgs_ref, const ImgPyr &imgs_cur, const cv::Size win_size,
                           const std::vector<cv::Point2f> &pts_ref, std::vector<cv::Point2f> &pts_cur,
                           std::vector<bool> &status, bool track_forward)
{
    const int N = pts_ref.size();
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    const int border = 8;
    const int x_min = border;
    const int y_min = border;
    const int x_max = imgs_ref[0].cols - border;
    const int y_max = imgs_cur[0].rows - border;

    std::vector<float> error;
    std::vector<uchar> status_forward;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
    //! forward track
    cv::calcOpticalFlowPyrLK(imgs_ref, imgs_cur, pts_ref, pts_cur, status_forward, error,
                             win_size, 3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    status.resize(N, false);
    std::fill(status.begin(), status.end(), false);
    for(int i = 0; i < N; ++i)
    {
        const cv::Point2f &pt = pts_cur[i];
        if(!status_forward[i] || pt.x < x_min || pt.y < y_min || pt.x > x_max || pt.y > y_max)
        {
            pts_cur[i] = cv::Point2f(0,0);
        }
        else
        {
            pts_cur[i] = pt;
            status[i] = true;
        }

    }

    if(!track_forward)
        return;

    std::vector<cv::Point2f> pts_cur_tracked;
    std::vector<cv::Point2f> pts_ref_tracked;
    std::vector<int> inlier_ids;
    pts_cur_tracked.reserve(N);
    inlier_ids.reserve(N);
    for(int i = 0; i < N; ++i)
    {
        if(!status[i])
            continue;

        pts_cur_tracked.push_back(pts_cur[i]);
        pts_ref_tracked.push_back(pts_ref[i]);
        inlier_ids.push_back(i);
    }

    //! backward track
    std::vector<uchar> status_back;
    cv::calcOpticalFlowPyrLK(imgs_cur, imgs_ref, pts_cur_tracked, pts_ref_tracked, status_back, error,
                             win_size, 3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    const int N1 = inlier_ids.size();
    for(int i = 0; i < N1; ++i)
    {
        const int idx = inlier_ids[i];
        const cv::Point2f &pt_real = pts_ref[idx];
        const cv::Point2f &pt_estm = pts_ref_tracked[i];
        const cv::Point2f delta = pt_real-pt_estm;
        if(!status_back[i] || (delta.x*delta.x + delta.y*delta.y) > 2.0)
        {
            status[idx] = false;
            pts_cur[idx] = cv::Point2f(0,0);
        }
    }

}

void Initializer::calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                                const std::vector<bool> &mask, std::vector<std::pair<int, float> >& disparities)
{
    const int N = pts1.size();
    assert(N == pts2.size());

    const size_t out_size= std::count(mask.begin(), mask.end(), true);
    disparities.clear();
    disparities.reserve(out_size);
    for(int i = 0; i < N; i++)
    {
        if(!mask[i])
            continue;

        float dx = pts2[i].x - pts1[i].x;
        float dy = pts2[i].y - pts1[i].y;
        disparities.emplace_back(std::make_pair(i, std::sqrt(dx*dx + dy*dy)));
    }
}

bool Initializer::findBestRT(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t,
                             const Matrix3d& K1, const Matrix3d& K2,
                             const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2,
                             cv::Mat& mask, std::vector<Vector3d>& P3Ds,
                             Matrix<double, 3, 4>& T)
{
    const size_t N = fts1.size();
    assert(N == fts2.size());
    if(mask.empty())
        mask = cv::Mat(fts1.size(), 1, CV_8UC1, cv::Scalar(255));

    //! P = K[R|t]
    MatrixXd P0(3, 4), P1(3, 4), P2(3, 4), P3(3, 4), P4(3, 4);
    MatrixXd T1(3, 4), T2(3, 4), T3(3, 4), T4(3, 4);

    //! P0 = K[I|0]
    P0 = K1 * MatrixXd::Identity(3,4);

    //! P1 = K[R1|t]
    T1.block(0,0,3,3) = R1; T1.col(3) = t;
    P1 = K2 * T1;

    //! P2 = K[R2|t]
    T2.block(0,0,3,3) = R2; T2.col(3) = t;
    P2 = K2 * T2;

    //! P3 = K[R1|-t]
    T3.block(0,0,3,3) = R1; T3.col(3) = -t;
    P3 = K2 * T3;

    //! P4 = K[R2|-t]
    T4.block(0,0,3,3) = R2; T4.col(3) = -t;
    P4 = K2 * T4;

    //! Do the cheirality check, and remove points too far away
    const double max_dist = 50.0;
    std::vector<Vector3d> P3Ds1(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds2(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds3(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds4(N, Vector3d(0,0,0));
    cv::Mat mask1 = mask.t();
    cv::Mat mask2 = mask.t();
    cv::Mat mask3 = mask.t();
    cv::Mat mask4 = mask.t();
    const uchar* mask_ptr = mask.ptr<uchar>(0);
    uchar* mask1_ptr = mask1.ptr<uchar>(0);
    uchar* mask2_ptr = mask2.ptr<uchar>(0);
    uchar* mask3_ptr = mask3.ptr<uchar>(0);
    uchar* mask4_ptr = mask4.ptr<uchar>(0);
    int nGood0 = cv::countNonZero(mask);
    int nGood1 = 0;
    int nGood2 = 0;
    int nGood3 = 0;
    int nGood4 = 0;

    Vector4d P3D;
    Vector3d T_P3D;
    for(size_t i = 0; i < N; ++i)
    {
        if(!mask_ptr[i])
            continue;

        const cv::Point2d ft1 = fts1[i];
        const cv::Point2d ft2 = fts2[i];

        //! P0 & P1
        triangulate(P0, P1, ft1, ft2, P3D);
        P3Ds1[i] = P3D.head(3);
        T_P3D = T1 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask1_ptr[i] = 0;
        else
            nGood1++;

        //! P0 & P2
        triangulate(P0, P2, ft1, ft2, P3D);
        P3Ds2[i] = P3D.head(3);
        T_P3D = T2 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask2_ptr[i] = 0;
        else
            nGood2++;

        //! P0 & P3
        triangulate(P0, P3, ft1, ft2, P3D);
        P3Ds3[i] = P3D.head(3);
        T_P3D = T3 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask3_ptr[i] = 0;
        else
            nGood3++;

        //! P0 & P4
        triangulate(P0, P4, ft1, ft2, P3D);
        P3Ds4[i] = P3D.head(3);
        T_P3D = T4 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask4_ptr[i] = 0;
        else
            nGood4++;
    }

    int maxGood = MAX(MAX(nGood1, nGood2), MAX(nGood3, nGood4));
    //! for normal, other count is 0
    if(maxGood < 0.9*nGood0)
    {
        LOG(WARNING) << "[INIT] Inliers: " << maxGood << ", less than 90% inliers after cheirality check!!!";
        return false;
    }

    if(maxGood == nGood1)
    {
        mask = mask1.t();
        P3Ds = P3Ds1;
        T = T1;
    }
    else if(maxGood == nGood2)
    {
        mask = mask2.t();
        P3Ds = P3Ds2;
        T = T2;
    }
    else if(maxGood == nGood3)
    {
        mask = mask3.t();
        P3Ds = P3Ds3;
        T = T3;
    }
    else if(maxGood == nGood4)
    {
        mask = mask4.t();
        P3Ds = P3Ds4;
        T = T4;
    }

    return true;
}

int Initializer::checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur,
                                   const  std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                                   const Matrix<double, 3, 4>& T, cv::Mat& mask, std::vector<Vector3d>& p3ds,
                                   const double sigma2)
{
    const int size = pts_ref.size();

    int inliers_count = 0;
    uchar* mask_ptr = mask.ptr<uchar>(0);

    for(int i = 0; i < size; ++i)
    {
        if(!mask_ptr[i])
            continue;

        const Vector3d p3d1 = p3ds[i];
        const double X = p3d1[0];
        const double Y = p3d1[1];
        const double Z = p3d1[2];
        double x1 = X / Z;
        double y1 = Y / Z;
        double dx1 = fts_ref[i].x - x1;
        double dy1 = fts_ref[i].y - y1;
        double err1 = dx1*dx1 + dy1*dy1;

        if(err1 > sigma2)
        {
            mask_ptr[i] = 0;
            continue;
        }

        Vector3d p3d2 = T * Vector4d(X,Y,Z,1);
        double x2 = p3d2[0] / p3d2[2];
        double y2 = p3d2[1] / p3d2[2];
        double dx2 = fts_cur[i].x - x2;
        double dy2 = fts_cur[i].y - y2;
        double err2 = dx2*dx2 + dy2*dy2;

        if(err2 > sigma2)
        {
            mask_ptr[i] = 0;
            continue;
        }

        inliers_count++;
        p3ds[i] = Vector3d(X, Y, Z);
    }

    return inliers_count;
}

void Initializer::triangulate(const Matrix<double, 3, 4>& P1, const Matrix<double, 3, 4>& P2,
                              const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D)
{
    MatrixXd A(4,4);
    A.row(0) = ft1.x*P1.row(2)-P1.row(0);
    A.row(1) = ft1.y*P1.row(2)-P1.row(1);
    A.row(2) = ft2.x*P2.row(2)-P2.row(0);
    A.row(3) = ft2.y*P2.row(2)-P2.row(1);

    JacobiSVD<MatrixXd> svd(A, ComputeThinV);
    MatrixXd V = svd.matrixV();

    P3D = V.col(3);
    P3D = P3D/P3D(3);
}

void Initializer::reduceVecor(std::vector<cv::Point2f>& pts, const cv::Mat& inliers)
{
    assert(inliers.cols == 1 || inliers.rows == 1);
    assert(inliers.type() == CV_8UC1);
    size_t size = MAX(inliers.cols, inliers.rows);
    assert(size == pts.size());

    std::vector<cv::Point2f>::iterator pts_iter = pts.begin();
    cv::Mat inliers_mat = inliers.clone();
    uchar* inliers_ptr = inliers_mat.ptr<uchar>(0);
    for(;pts_iter!=pts.end();)
    {
        if(!(*inliers_ptr))
        {
            *inliers_ptr = inliers_mat.data[--size];
            *pts_iter = pts.back();
            pts.pop_back();
            continue;
        }
        inliers_ptr++;
        pts_iter++;
    }
}

void Initializer::reduceVecor(std::vector<cv::Point2d>& fts, const cv::Mat& inliers)
{
    assert(inliers.cols == 1 || inliers.rows == 1);
    assert(inliers.type() == CV_8UC1);
    size_t size = MAX(inliers.cols, inliers.rows);
    assert(size == fts.size());

    std::vector<cv::Point2d>::iterator fts_iter = fts.begin();
    cv::Mat inliers_mat = inliers.clone();
    uchar* inliers_ptr = inliers_mat.ptr<uchar>(0);
    for(;fts_iter!=fts.end();)
    {
        if(!(*inliers_ptr))
        {
            *inliers_ptr = inliers_mat.data[--size];
            *fts_iter =fts.back();
            fts.pop_back();
            continue;
        }
        inliers_ptr++;
        fts_iter++;
    }
}

int Fundamental::findFundamentalMat(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d &F,
                                    std::vector<bool> &inliers, double sigma2, int max_iterations, const bool bE)
{
    assert(fts_prev.size() == fts_next.size());

    return runRANSAC(fts_prev, fts_next, F, inliers, sigma2, max_iterations, bE);
}

void Fundamental::run8point(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d& F, const bool bE)
{
    const int N = fts_prev.size();
    assert(N >= 8);

    std::vector<cv::Point2d> fts_prev_norm;
    std::vector<cv::Point2d> fts_next_norm;
    Matrix3d T1, T2;
    Normalize(fts_prev, fts_prev_norm, T1);
    Normalize(fts_next, fts_next_norm, T2);

    MatrixXd A(N,9);
    for(int i = 0; i < N; ++i)
    {
        const double u1 = fts_prev_norm[i].x;
        const double v1 = fts_prev_norm[i].y;
        const double u2 = fts_next_norm[i].x;
        const double v2 = fts_next_norm[i].y;

        A.row(i) <<  u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1;
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullV);
    MatrixXd Va = svd.matrixV();

    VectorXd Fv = Va.col(8);
    Matrix3d Ft(Fv.data());

    MatrixXd eF = T2.transpose()*Ft.transpose()*T1;

    JacobiSVD<MatrixXd> svd1(eF, ComputeFullV|ComputeFullU);
    MatrixXd V = svd1.matrixV();
    MatrixXd U = svd1.matrixU();
    Vector3d S = svd1.singularValues();
    if(bE)
    {
        const double s = 0.5*(S(0)+S(1));
        S(0) = s;
        S(1) = s;
    }
    S(2) = 0;
    DiagonalMatrix<double, Dynamic> W(S);

    F = U * W * V.transpose();

    double F22 = F(2, 2);
    if(fabs(F22) > std::numeric_limits<double>::epsilon())
        F /= F22;
}


int Fundamental::runRANSAC(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d& F,
                           std::vector<bool> &inliers, const double sigma2, const int max_iterations, const bool bE)
{
    const int N = fts_prev.size();
    const int modelPoints = 8;

    const double threshold = 3.841*sigma2;
    const int max_iters = MIN(MAX(max_iterations, 1), 1000);

    std::vector<int> total_points;
    total_points.reserve(N);
    for(int i = 0; i < N; ++i)
    {
        if(inliers[i])
            total_points.push_back(i);
    }
    const int npoints = total_points.size();
    assert(npoints >= modelPoints);

    std::vector<cv::Point2d> fts1(modelPoints);
    std::vector<cv::Point2d> fts2(modelPoints);
    std::vector<cv::Point2d> fts1_norm;
    std::vector<cv::Point2d> fts2_norm;
    Matrix3d F_temp;
    int max_inliers = 0;
    int niters = max_iters;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<int> points = total_points;
        for(int i = 0; i < modelPoints; ++i)
        {
            int randi = Rand(0, points.size()-1);
            fts1[i] = fts_prev[points[randi]];
            fts2[i] = fts_next[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        run8point(fts1, fts2, F_temp, bE);

        int inliers_count = 0;
        std::vector<bool> inliers_temp(N, false);
        for(const int id : total_points)
        {
            double error1, error2;
            computeErrors(fts_prev[id], fts_next[id], F_temp, error1, error2);

            const double error = MAX(error1, error2);

            if(error < threshold)
            {
                inliers_temp[id] = true;
                inliers_count++;
            }
        }

        if(inliers_count > max_inliers)
        {
            max_inliers = inliers_count;
            inliers = inliers_temp;

            if(inliers_count < npoints)
            {
                //! N = log(1-p)/log(1-omega^s)
                //! p = 99%
                //! number of set: s = 8
                //! omega = inlier points / total points
                const static double num = log(1 - 0.99);
                const double omega = inliers_count*1.0 / npoints;
                const double denom = log(1 - pow(omega, modelPoints));

                niters = (denom >= 0 || -num >= max_iters*(-denom)) ? max_iters : round(num / denom);
            }
            else
                break;
        }

    }//! iterations

    fts1.clear();
    fts2.clear();
    for(int n = 0; n < N; ++n)
    {
        if(!inliers[n])
            continue;

        fts1.push_back(fts_prev[n]);
        fts2.push_back(fts_next[n]);
    }

    run8point(fts1, fts2, F, bE);

    return max_inliers;
}

void Fundamental::Normalize(const std::vector<cv::Point2d>& fts, std::vector<cv::Point2d>& fts_norm, Matrix3d& T)
{
    const int N = fts.size();
    if(N == 0)
        return;

    fts_norm.resize(N);

    cv::Point2d mean(0,0);
    for(int i = 0; i < N; ++i)
    {
        mean += fts[i];
    }
    mean = mean/N;

    cv::Point2d mean_dev(0,0);

    for(int i = 0; i < N; ++i)
    {
        fts_norm[i] = fts[i] - mean;

        mean_dev.x += fabs(fts_norm[i].x);
        mean_dev.y += fabs(fts_norm[i].y);
    }
    mean_dev /= N;

    const double scale_x = 1.0/mean_dev.x;
    const double scale_y = 1.0/mean_dev.y;

    for(int i=0; i<N; i++)
    {
        fts_norm[i].x *= scale_x;
        fts_norm[i].y *= scale_y;
    }

    T <<  scale_x, 0, -mean.x*scale_x, 0, scale_y, -mean.y*scale_y, 0,0,1;
}

inline void Fundamental::computeErrors(const cv::Point2d& p1, const cv::Point2d& p2, Matrix3d& F, double& err1, double& err2)
{
    const double &F0 = F(0,0);
    const double &F1 = F(0,1);
    const double &F2 = F(0,2);
    const double &F3 = F(1,0);
    const double &F4 = F(1,1);
    const double &F5 = F(1,2);
    const double &F6 = F(2,0);
    const double &F7 = F(2,1);
    const double &F8 = F(2,2);

    //! point X1 = (u1, v1, 1)^T in first image
    //! poInt X2 = (u2, v2, 1)^T in second image
    const double u1 = p1.x;
    const double v1 = p1.y;
    const double u2 = p2.x;
    const double v2 = p2.y;

    //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
    const double a2 = F0*u1 + F1*v1 + F2;
    const double b2 = F3*u1 + F4*v1 + F5;
    const double c2 = F6*u1 + F7*v1 + F8;
    //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
    const double a1 = F0*u2 + F3*v2 + F6;
    const double b1 = F1*u2 + F4*v2 + F7;
    const double c1 = F2*u2 + F5*v2 + F8;

    //! distance from point to line: d^2 = |ax+by+c|^2/(a^2+b^2)
    //! X2 to L2 in second image
    const double dist2 = a2*u2 + b2*v2 + c2;
    const double square_dist2 = dist2*dist2/(a2*a2 + b2*b2);
    //! X1 to L1 in first image
    const double dist1 = a1*u1 + b1*v1 + c1;
    const double square_dist1 = dist1*dist1/(a1*a1 + b1*b1);

    err1 = square_dist1;
    err2 = square_dist2;
}

void Fundamental::decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t)
{
    JacobiSVD<MatrixXd> svd(E, ComputeFullV|ComputeFullU);
    MatrixXd V = svd.matrixV();
    MatrixXd U = svd.matrixU();

    if(U.determinant() < 0) U *= -1.;
    if(V.determinant() < 0) V *= -1.;

    Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    MatrixXd Vt = V.transpose();
    R1 = U * W * Vt;
    R2 = U * W.transpose() * Vt;

    t = U.col(2);
    t = t / t.norm();
}

}