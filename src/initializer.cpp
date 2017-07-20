#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
#include "global.hpp"

#include "initializer.hpp"
#include <opencv/cxeigen.hpp>

namespace ssvo{

Initializer::Initializer(const cv::Mat& K, const cv::Mat& D)
{
    finished_ = false;
    K_ = K.clone();
    D_ = D.clone();
}

InitResult Initializer::addFirstImage(const cv::Mat& img_ref, std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& fts)
{
    assert(pts.size() == fts.size());
    //! reset
    pts_ref_.clear();
    fts_ref_.clear();
    pts_cur_.clear();
    fts_cur_.clear();
    p3ds_.clear();
    disparities_.clear();
    inliers_.release();
    finished_ = false;

    //! check corner number of first image
    if(pts.size() < Config::initMinCorners())
    {
        SSVO_WARN_STREAM("[INIT] First image has too less corners !!!");
        return RESET;
    }

    //! get refrence image
    img_ref_ = img_ref.clone();
    //! set initial flow
    pts_ref_.resize(pts.size());
    fts_ref_.resize(pts.size());
    pts_cur_.resize(pts.size());
    std::copy(pts.begin(), pts.end(), pts_ref_.begin());
    std::copy(fts.begin(), fts.end(), fts_ref_.begin());
    std::copy(pts.begin(), pts.end(), pts_cur_.begin());

    return SUCCESS;
}

InitResult Initializer::addSecondImage(const cv::Mat& img_cur)
{
    if(finished_)
    {
        SSVO_WARN_STREAM("[INIT] Last initialization is succeed! Plesase reset!")
        return RESET;
    }

    img_cur_ = img_cur;
    double t1 = (double)cv::getTickCount();
    //! [1] KLT tracking
    kltTrack(img_ref_, img_cur_, pts_ref_, pts_cur_, inliers_);
    reduceVecor(pts_ref_, inliers_);
    reduceVecor(pts_cur_, inliers_);
    reduceVecor(fts_ref_, inliers_);
    inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));

    //! calculate disparities on undistorted points
    cv::undistortPoints(pts_cur_, fts_cur_, K_, D_);
    calcDisparity(pts_ref_, pts_cur_, disparities_);

    SSVO_INFO_STREAM("[INIT] KLT tracking points: " << disparities_.size());
    if(disparities_.size() < Config::initMinTracked()) return RESET;

    double t2 = (double)cv::getTickCount();
    std::vector<float> disparities_temp = disparities_;
    std::sort(disparities_temp.begin(), disparities_temp.end());
    float disparity = disparities_temp.at(disparities_temp.size()/2);
    float max_disparity = disparities_temp.at(disparities_temp.size()*3/4)*2;
    uchar* inliers_ptr = inliers_.ptr<uchar>(0);
    bool reduce = false;
    for (int i = 0; i < pts_ref_.size(); ++i)
    {
        if(disparities_[i] > max_disparity)
        {
            reduce = true;
            inliers_ptr[i] = 0;
        }
    }
    if(reduce)
    {
        reduceVecor(pts_ref_, inliers_);
        reduceVecor(pts_cur_, inliers_);
        reduceVecor(fts_ref_, inliers_);
        reduceVecor(fts_cur_, inliers_);
        inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));
    }

    SSVO_INFO_STREAM("[INIT] Avage disparity: " << disparity);
    if(disparity < Config::initMinDisparity()) return FAILURE;

    double t3 = (double)cv::getTickCount();
    //! [2] geometry check by F matrix
    //cv::Mat F = cv::findFundamentalMat(fts_ref_, fts_cur_, inliers_, cv::FM_RANSAC);
    //F.convertTo(F, CV_32FC1);
    //int inliers_count = cv::countNonZero(inliers_);
    cv::Mat F;
    int inliers_count = Fundamental::findFundamentalMat(fts_ref_, fts_cur_, F, inliers_, Config::initUnSigma(), Config::initMaxRansacIters());
    if(inliers_count!= pts_ref_.size())
    {
        reduceVecor(pts_ref_, inliers_);
        reduceVecor(pts_cur_, inliers_);
        reduceVecor(fts_ref_, inliers_);
        reduceVecor(fts_cur_, inliers_);
        inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));
    }
    SSVO_INFO_STREAM("[INIT] Inliers after Fundamental Maxtrix RANSCA check: " << inliers_count);
    if(inliers_count < Config::initMinInliers()) return FAILURE;

    double t4 = (double)cv::getTickCount();
    cv::Mat R1, R2, t;
    cv::Mat E = K_.t() * F * K_;
    Fundamental::decomposeEssentialMat(E, R1, R2, t);

    double t5 = (double)cv::getTickCount();
    cv::Mat T;
    cv::Mat P3Ds;
    cv::Mat K =cv::Mat::eye(3,3,CV_32FC1);
    //! [3] cheirality check
    bool succeed = findBestRT(R1, R2, t, K, K, fts_ref_, fts_cur_, inliers_, P3Ds, T);
    if(!succeed) return FAILURE;
    SSVO_INFO_STREAM("[INIT] Inliers after cheirality check: " << cv::countNonZero(inliers_));

    //! [4] reprojective check
    succeed = checkReprejectErr(pts_ref_, pts_cur_, fts_ref_, fts_cur_, T, inliers_, P3Ds, Config::initUnSigma()*8, p3ds_);
    if(!succeed) return FAILURE;

    double t6 = (double)cv::getTickCount();
    std::cout << "Time: " << (t2-t1)/cv::getTickFrequency() << " "
                          << (t3-t2)/cv::getTickFrequency() << " "
                          << (t4-t3)/cv::getTickFrequency() << " "
                          << (t5-t4)/cv::getTickFrequency() << " "
                          << (t6-t5)/cv::getTickFrequency() << std::endl;

    SSVO_INFO_STREAM("[INIT] Initialization succeed!");

    finished_ = true;
    return SUCCESS;

}

void Initializer::getUndistInilers(std::vector<cv::Point2f>& fts_ref, std::vector<cv::Point2f>& fts_cur) const
{
    if(!finished_)
    {
        SSVO_WARN_STREAM("[INIT] Please waiting until inintialization finished!");
        return;
    }

    fts_ref = fts_ref_;
    fts_cur = fts_cur_;
}
void Initializer::getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const
{
    pts_ref = pts_ref_;
    pts_cur = pts_cur_;
}

void Initializer::kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur, std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, cv::Mat& inliers)
{
    const int klt_win_size = 21;
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    const int border = 8;
    const int x_min = border;
    const int y_min = border;
    const int x_max = img_ref.cols - border;
    const int y_max = img_ref.rows - border;

    std::vector<float> error;
    std::vector<uchar> status;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);

    cv::calcOpticalFlowPyrLK(
        img_ref, img_cur,
        pts_ref, pts_cur,
        status, error,
        cv::Size(klt_win_size, klt_win_size), 3,
        termcrit, cv::OPTFLOW_USE_INITIAL_FLOW
    );

    std::vector<cv::Point2f>::iterator pts_cur_it = pts_cur.begin();
    std::vector<cv::Point2f>::iterator pts_cur_end = pts_cur.end();

    inliers = cv::Mat(pts_cur.size(), 1, CV_8UC1, cv::Scalar(255));
    uchar* inliers_ptr = inliers.ptr<uchar>(0);
    for(int i = 0; pts_cur_it!= pts_cur_end; ++i, ++pts_cur_it)
    {
        if(!status[i] || pts_cur_it->x < x_min || pts_cur_it->y < y_min || pts_cur_it->x > x_max || pts_cur_it->y > y_max)
        {
            inliers_ptr[i] = 0;
            continue;
        }
    }
}

void Initializer::calcDisparity(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, std::vector<float>& disparities)
{
    std::vector<cv::Point2f>::iterator pts1_it = pts1.begin();
    std::vector<cv::Point2f>::iterator pts2_it = pts2.begin();

    disparities.clear();
    disparities.reserve(pts1.size());
    for(std::vector<cv::Point2f>::iterator it_end = pts1.end(); pts1_it != it_end; pts1_it++, pts2_it++)
    {
        float dx = pts2_it->x - pts1_it->x;
        float dy = pts2_it->y - pts1_it->y;
        disparities.push_back(sqrt(dx*dx + dy*dy));
    }
}

bool Initializer::findBestRT(const cv::Mat& R1, const cv::Mat& R2, const cv::Mat& t, const cv::Mat& K1, const cv::Mat& K2,
                             const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3Ds, cv::Mat& T)
{
    assert(pts1.size() == pts2.size());
    if(mask.empty())
        mask = cv::Mat(pts1.size(), 1, CV_8UC1, cv::Scalar(255));

    //! P = K[R|t]
    cv::Mat P0 = cv::Mat::eye(3, 4, R1.type());
    cv::Mat T1(3, 4, R1.type(), cv::Scalar(0)), T2(3, 4, R1.type(), cv::Scalar(0)), T3(3, 4, R1.type(), cv::Scalar(0)), T4(3, 4, R1.type(), cv::Scalar(0));
    cv::Mat P1(3, 4, R1.type(), cv::Scalar(0)), P2(3, 4, R1.type(), cv::Scalar(0)), P3(3, 4, R1.type(), cv::Scalar(0)), P4(3, 4, R1.type(), cv::Scalar(0));
    cv::Mat T_P3Ds;

    //! P0 = K[I|0]
    K1.copyTo(P0.rowRange(0,3).colRange(0,3));

    //! P1 = K[R1|t]
    T1.rowRange(0,3).colRange(0,3) = R1 * 1.0;
    T1.rowRange(0,3).col(3) = t * 1.0;
    P1 = K2*T1;

    //! P2 = K[R2|t]
    T2.rowRange(0,3).colRange(0,3) = R2 * 1.0;
    T2.rowRange(0,3).col(3) = t * 1.0;
    P2 = K2*T2;

    //! P3 = K[R1|-t]
    T3.rowRange(0,3).colRange(0,3) = R1 * 1.0;
    T3.rowRange(0,3).col(3) = t * -1.0;
    P3 = K2*T3;

    //! P4 = K[R2|-t]
    T4.rowRange(0,3).colRange(0,3) = R2 * 1.0;
    T4.rowRange(0,3).col(3) = t * -1.0;
    P4 = K2*T4;

    //! Do the cheirality check, and remove points too far away
    const float max_dist = 50.0;
    cv::Mat P3Ds1;
    triangulate(P0, P1, pts1, pts2, mask, P3Ds1);
    cv::Mat mask1 = mask.t();
    mask1 &= (P3Ds1.row(2) > 0) & (P3Ds1.row(2) < max_dist);
    T_P3Ds = T1*P3Ds1;
    mask1 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds2;
    triangulate(P0, P2, pts1, pts2, mask, P3Ds2);
    cv::Mat mask2 = mask.t();
    mask2 &= (P3Ds2.row(2) > 0) & (P3Ds2.row(2) < max_dist);
    T_P3Ds = T2*P3Ds2;
    mask2 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds3;
    triangulate(P0, P3, pts1, pts2, mask, P3Ds3);
    cv::Mat mask3 = mask.t();
    mask3 &= (P3Ds3.row(2) > 0) & (P3Ds3.row(2) < max_dist);
    T_P3Ds = T3*P3Ds3;
    mask3 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds4;
    triangulate(P0, P4, pts1, pts2, mask, P3Ds4);
    cv::Mat mask4 = mask.t();
    mask4 &= (P3Ds4.row(2) > 0) & (P3Ds4.row(2) < max_dist);
    T_P3Ds = T4*P3Ds4;
    mask4 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    int nGood0 = cv::countNonZero(mask);
    int nGood1 = cv::countNonZero(mask1);
    int nGood2 = cv::countNonZero(mask2);
    int nGood3 = cv::countNonZero(mask3);
    int nGood4 = cv::countNonZero(mask4);

    int maxGood = MAX(MAX(nGood1, nGood2), MAX(nGood3, nGood4));
    //! for normal, other count is 0
    if(maxGood < 0.9*nGood0)
    {
        SSVO_WARN_STREAM("[INIT] Inliers: " << maxGood << ", less than 90% inliers after cheirality check!!!");
        return false;
    }


    if(maxGood == nGood1)
    {
        mask = mask1.t();
        P3Ds = P3Ds1.clone();
        T = T1.clone();
    }
    else if(maxGood == nGood2)
    {
        mask = mask2.t();
        P3Ds = P3Ds2.clone();
        T = T2.clone();
    }
    else if(maxGood == nGood3)
    {
        mask = mask3.t();
        P3Ds = P3Ds3.clone();
        T = T3.clone();
    }
    else if(maxGood == nGood4)
    {
        mask = mask4.t();
        P3Ds = P3Ds4.clone();
        T = T4.clone();
    }

    return true;
}

bool Initializer::checkReprejectErr(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, std::vector<cv::Point2f>& fts_ref, std::vector<cv::Point2f>& fts_cur,
                       const cv::Mat& T, const cv::Mat& mask, const cv::Mat& P3Ds, const float sigma, std::vector<Vector3f>& p3ds)
{
    assert(T.type() == CV_32FC1);
    assert(P3Ds.type() == CV_32FC1);

    const int n = pts_ref_.size();
    const float sigma2 = sigma*sigma;
    p3ds_.clear(); p3ds_.reserve(n);

    std::vector<int> inliers;
    inliers.reserve(n);
    const uchar* mask_ptr = mask.ptr<uchar>(0);
    const float* P3Ds_ptr = P3Ds.ptr<float>(0);
    const int strick = P3Ds.cols;
    const int strick2 = strick*2;

    for(int i = 0; i < n; ++i)
    {
        if(!mask_ptr[i])
            continue;

        const float* p3d1 = &P3Ds_ptr[i];
        const float X = p3d1[0];
        const float Y = p3d1[strick];
        const float Z = p3d1[strick2];
        float x1 = X / Z;
        float y1 = Y / Z;
        float dx1 = fts_ref[i].x - x1;
        float dy1 = fts_ref[i].y - y1;
        float err1 = dx1*dx1 + dy1*dy1;

        if(err1 > sigma2)
            continue;

        cv::Mat P1 = (cv::Mat_<float>(4,1) <<X, Y, Z, 1);
        cv::Mat P2 = T*P1;
        float* p3d2 = P2.ptr<float>(0);
        float x2 = p3d2[0] / p3d2[2];
        float y2 = p3d2[1] / p3d2[2];
        float dx2 = fts_cur[i].x - x2;
        float dy2 = fts_cur[i].y - y2;
        float err2 = dx2*dx2 + dy2*dy2;

        if(err2 > sigma2)
            continue;

        inliers.push_back(i);
        p3ds.push_back(Vector3f(X, Y, Z));
    }
    std::cout << "  " << inliers.size() << std::endl;
    return false;
}

#if 0
void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3D)
{
    const int n = pts1.size();
    assert(n == pts2.size());
    assert(mask.cols == 1 && mask.rows == n);
    uchar *mask_ptr = mask.ptr<uchar>(0);
    P3D = cv::Mat::zeros(4, n, CV_32FC1);
    for(int i = 0; i < n; ++i)
    {
        cv::Mat wP;

        if(mask_ptr[i])
            triangulate(P1, P2, pts1[i], pts2[i], wP);
        else
            wP = cv::Mat::zeros(4,1,CV_32FC1);
        wP.copyTo(P3D.col(i));
    }
}
#else
void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3D)
{
    const int n = pts1.size();
    assert(n == pts2.size());
    assert(mask.cols == 1 && mask.rows == n);
    uchar *mask_ptr = mask.ptr<uchar>(0);
    MatrixXf eP1, eP2;
    cv::cv2eigen(P1, eP1);
    cv::cv2eigen(P2, eP2);
    MatrixXf eP3D = MatrixXf::Zero(4, n);
    P3D = cv::Mat::zeros(4, n, CV_32FC1);
    for(int i = 0; i < n; ++i)
    {
        Vector4f wP = Vector4f::Zero();

        if(mask_ptr[i]) {
            triangulate(eP1, eP2, pts1[i], pts2[i], wP);
        }

        eP3D.col(i) = wP;

    }
    cv::eigen2cv(eP3D, P3D);
}
#endif

void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2f& pt1, const cv::Point2f& pt2, cv::Mat& P3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = pt1.x*P1.row(2)-P1.row(0);
    A.row(1) = pt1.y*P1.row(2)-P1.row(1);
    A.row(2) = pt2.x*P2.row(2)-P2.row(0);
    A.row(3) = pt2.y*P2.row(2)-P2.row(1);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    P3D = Vt.row(3).t();
    P3D = P3D/P3D.at<float>(3);
}

void Initializer::triangulate(const MatrixXf& P1, const MatrixXf& P2, const cv::Point2f& pt1, const cv::Point2f& pt2, Vector4f& P3D)
{
    MatrixXf A(4,4);
    A.row(0) = pt1.x*P1.row(2)-P1.row(0);
    A.row(1) = pt1.y*P1.row(2)-P1.row(1);
    A.row(2) = pt2.x*P2.row(2)-P2.row(0);
    A.row(3) = pt2.y*P2.row(2)-P2.row(1);

    JacobiSVD<MatrixXf> svd(A, ComputeThinV);
    MatrixXf V = svd.matrixV();

    P3D = V.col(3);
    P3D = P3D/P3D(3);
}

void Initializer::reduceVecor(std::vector<cv::Point2f>& pts, const cv::Mat& inliers)
{
    assert(inliers.cols == 1 || inliers.rows == 1);
    assert(inliers.type() == CV_8UC1);
    int size = MAX(inliers.cols, inliers.rows);
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

int Fundamental::findFundamentalMat(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat &F,
                                    cv::Mat& inliers, float sigma, int max_iterations)
{
    assert(pts_prev.size() == pts_next.size());

    return runRANSAC(pts_prev, pts_next, F, inliers, sigma, max_iterations);
}

#if 1
void Fundamental::run8point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F)
{
    const int N = pts_prev.size();
    assert(N >= 8);

    std::vector<cv::Point2f> pts_prev_norm;
    std::vector<cv::Point2f> pts_next_norm;
    cv::Mat T1, T2;
    Normalize(pts_prev, pts_prev_norm, T1);
    Normalize(pts_next, pts_next_norm, T2);

    cv::Mat A(N, 9, CV_32F);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev_norm[i].x;
        const float v1 = pts_prev_norm[i].y;
        const float u2 = pts_next_norm[i].x;
        const float v2 = pts_next_norm[i].y;
        float* a = A.ptr<float>(i);

        a[0] = u2*u1;
        a[1] = u2*v1;
        a[2] = u2;
        a[3] = v2*u1;
        a[4] = v2*v1;
        a[5] = v2;
        a[6] = u1;
        a[7] = v1;
        a[8] = 1;
    }

    cv::Mat U, W, vt;

    cv::eigen(A.t()*A, W, vt);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, W, U, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    W.at<float>(2) = 0;

    cv::Mat F_norm = U*cv::Mat::diag(W)*vt;

    F = T2.t()*F_norm*T1;
    float F22 = F.at<float>(2, 2);
    if(fabs(F22) > FLT_EPSILON)
        F /= F22;
}
#else
void Fundamental::run8point(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F)
{
    const int N = pts_prev.size();
    assert(N >= 8);

    std::vector<cv::Point2f> pts_prev_norm;
    std::vector<cv::Point2f> pts_next_norm;
    Matrix3f T1, T2;
    Normalize(pts_prev, pts_prev_norm, T1);
    Normalize(pts_next, pts_next_norm, T2);

    MatrixXf A(N,9);
    for(int i = 0; i < N; ++i)
    {
        const float u1 = pts_prev_norm[i].x;
        const float v1 = pts_prev_norm[i].y;
        const float u2 = pts_next_norm[i].x;
        const float v2 = pts_next_norm[i].y;

        A.row(i) <<  u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1;
    }

    JacobiSVD<MatrixXf> svd(A, ComputeFullV);
    MatrixXf Va = svd.matrixV();

    VectorXf Fv = Va.col(8);
    Matrix3f Ft(Fv.data());

    JacobiSVD<MatrixXf> svd1(Ft.transpose(), ComputeFullV|ComputeFullU);
    MatrixXf V = svd1.matrixV();
    MatrixXf U = svd1.matrixU();
    Vector3f S = svd1.singularValues();
    S(2) = 0;
    DiagonalMatrix<float, Dynamic> W(S);

    Matrix3f Fn = U * W * V.transpose();

    MatrixXf FF = T2.transpose()*Fn*T1;
    float F22 = FF(2, 2);
    if(fabs(F22) > FLT_EPSILON)
        FF /= F22;

    cv::eigen2cv(FF, F);

}
#endif

int Fundamental::runRANSAC(const std::vector<cv::Point2f>& pts_prev, const std::vector<cv::Point2f>& pts_next, cv::Mat& F, cv::Mat& inliers, float sigma, int max_iterations)
{
    const int N = pts_prev.size();
    const int modelPoints = 8;
    assert(N >= modelPoints);

    const double threshold = 3.841*sigma*sigma;
    const int max_iters = MIN(MAX(max_iterations, 1), 1000);

    std::vector<int> total_points;
    for(int i = 0; i < N; ++i)
    {
        total_points.push_back(i);
    }

    std::vector<cv::Point2f> pts1(modelPoints);
    std::vector<cv::Point2f> pts2(modelPoints);
    std::vector<cv::Point2f> pts1_norm;
    std::vector<cv::Point2f> pts2_norm;
    cv::Mat F_temp;
    inliers = cv::Mat::zeros(N, 1, CV_8UC1);
    int max_inliers = 0;
    int niters = max_iters;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<int> points = total_points;
        for(int i = 0; i < modelPoints; ++i)
        {
            int randi = Rand(0, points.size()-1);
            pts1[i] = pts_prev[points[randi]];
            pts2[i] = pts_next[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        run8point(pts1, pts2, F_temp);

        int inliers_count = 0;
        cv::Mat inliers_temp = cv::Mat::zeros(N, 1, CV_8UC1);
        uchar* inliers_ptr = inliers_temp.ptr<uchar>(0);
        for(int n = 0; n < N; ++n)
        {
            float error1, error2;
            computeErrors(pts_prev[n], pts_next[n], F_temp.ptr<float>(0), error1, error2);

            const float error = MAX(error1, error2);

            if(error < threshold)
            {
                inliers_ptr[n] = 0xff;
                //inliers_temp.at<uchar>(n, 0) = 255;
                inliers_count++;
            }
        }

        if(inliers_count > max_inliers)
        {
            max_inliers = inliers_count;
            inliers = inliers_temp.clone();

            if(inliers_count < N)
            {
                //! N = log(1-p)/log(1-omega^s)
                //! p = 99%
                //! number of set: s = 8
                //! omega = inlier points / total points
                const static double num = log(1 - 0.99);
                const double omega = inliers_count*1.0 / N;
                const double denom = log(1 - pow(omega, modelPoints));

                niters = (denom >= 0 || -num >= max_iters*(-denom)) ? max_iters : round(num / denom);
            }
            else
                break;
        }

    }//! iterations

    pts1.clear();
    pts2.clear();
    for(int n = 0; n < N; ++n)
    {
        if(0 == inliers.at<uchar>(n, 0))
        {
            continue;
        }

        pts1.push_back(pts_prev[n]);
        pts2.push_back(pts_next[n]);
    }

    run8point(pts1, pts2, F);

    return max_inliers;
}

void Fundamental::Normalize(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_norm, cv::Mat& T)
{
    const int N = pts.size();
    if(N == 0)
        return;

    pts_norm.resize(N);

    cv::Point2f mean(0,0);
    for(int i = 0; i < N; ++i)
    {
        mean += pts[i];
    }
    mean = mean/N;

    cv::Point2f mean_dev(0,0);

    for(int i = 0; i < N; ++i)
    {
        pts_norm[i] = pts[i] - mean;

        mean_dev.x += fabs(pts_norm[i].x);
        mean_dev.y += fabs(pts_norm[i].y);
    }
    mean_dev /= N;

    const float scale_x = 1.0/mean_dev.x;
    const float scale_y = 1.0/mean_dev.y;

    for(int i=0; i<N; i++)
    {
        pts_norm[i].x *= scale_x;
        pts_norm[i].y *= scale_y;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = scale_x;
    T.at<float>(1,1) = scale_y;
    T.at<float>(0,2) = -mean.x*scale_x;
    T.at<float>(1,2) = -mean.y*scale_y;
}

void Fundamental::Normalize(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_norm, Matrix3f& T)
{
    const int N = pts.size();
    if(N == 0)
        return;

    pts_norm.resize(N);

    cv::Point2f mean(0,0);
    for(int i = 0; i < N; ++i)
    {
        mean += pts[i];
    }
    mean = mean/N;

    cv::Point2f mean_dev(0,0);

    for(int i = 0; i < N; ++i)
    {
        pts_norm[i] = pts[i] - mean;

        mean_dev.x += fabs(pts_norm[i].x);
        mean_dev.y += fabs(pts_norm[i].y);
    }
    mean_dev /= N;

    const float scale_x = 1.0/mean_dev.x;
    const float scale_y = 1.0/mean_dev.y;

    for(int i=0; i<N; i++)
    {
        pts_norm[i].x *= scale_x;
        pts_norm[i].y *= scale_y;
    }

    T <<  scale_x, 0, -mean.x*scale_x, 0, scale_y, -mean.y*scale_y, 0,0,1;
}

inline void Fundamental::computeErrors(const cv::Point2f& p1, const cv::Point2f& p2, const float* F, float& err1, float& err2)
{
    //! point X1 = (u1, v1, 1)^T in first image
    //! poInt X2 = (u2, v2, 1)^T in second image
    const float u1 = p1.x;
    const float v1 = p1.y;
    const float u2 = p2.x;
    const float v2 = p2.y;

    //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
    const float a2 = F[0]*u1 + F[1]*v1 + F[2];
    const float b2 = F[3]*u1 + F[4]*v1 + F[5];
    const float c2 = F[6]*u1 + F[7]*v1 + F[8];
    //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
    const float a1 = F[0]*u2 + F[3]*v2 + F[6];
    const float b1 = F[1]*u2 + F[4]*v2 + F[7];
    const float c1 = F[2]*u2 + F[5]*v2 + F[8];

    //! distance from point to line: d^2 = |ax+by+c|^2/(a^2+b^2)
    //! X2 to L2 in second image
    const float dist2 = a2*u2 + b2*v2 + c2;
    const float square_dist2 = dist2*dist2/(a2*a2 + b2*b2);
    //! X1 to L1 in first image
    const float dist1 = a1*u1 + b1*v1 + c1;
    const float square_dist1 = dist1*dist1/(a1*a1 + b1*b1);

    err1 = square_dist1;
    err2 = square_dist2;
}

//! modified from OpenCV
void Fundamental::decomposeEssentialMat(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t)
{
    assert(E.cols == 3 && E.rows == 3);

    cv::Mat U, D,Vt;
    cv::SVD::compute(E, D, U, Vt);

    if(determinant(U) < 0) U *= -1.;
    if(determinant(Vt) < 0) Vt *= -1.;

    cv::Mat W = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;

    U.col(2).copyTo(t);
    t = t / cv::norm(t);
}

}