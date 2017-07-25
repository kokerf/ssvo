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

InitResult Initializer::addFirstImage(const cv::Mat& img_ref, std::vector<cv::Point2f>& pts, std::vector<cv::Point2d>& fts)
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

    SSVO_INFO_STREAM("[INIT] ~~ Processing second image ~~ ");

    double t1 = (double)cv::getTickCount();
    //! [1] KLT tracking
    kltTrack(img_ref_, img_cur_, pts_ref_, pts_cur_, inliers_);
//    reduceVecor(pts_ref_, inliers_);
//    reduceVecor(pts_cur_, inliers_);
//    reduceVecor(fts_ref_, inliers_);
//    inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));
    calcDisparity(pts_ref_, pts_cur_, inliers_, disparities_);
    SSVO_INFO_STREAM("[INIT] KLT tracking points: " << disparities_.size());
    if(disparities_.size() < Config::initMinTracked()) return RESET;

    double t2 = (double)cv::getTickCount();
    //! [2] calculate disparities
    std::vector<std::pair<int, float> > disparities_temp = disparities_;
    std::sort(disparities_temp.begin(), disparities_temp.end(), [](const std::pair<int, float> disp1, const std::pair<int, float> disp2){return (disp1.second < disp2.second);});
    float disparity = disparities_temp.at(disparities_temp.size()/2).second;
    float max_disparity = disparities_temp.at(disparities_temp.size()*3/4).second * 2;
    uchar* inliers_ptr = inliers_.ptr<uchar>(0);
    //bool reduce = false;
    for (int i = 0; i < disparities_.size(); ++i)
    {
        if(disparities_[i].second > max_disparity)
        {
            const int id = disparities_[i].first;
            inliers_ptr[id] = 0;
        }
    }
//    if(reduce)
//    {
//        reduceVecor(pts_ref_, inliers_);
//        reduceVecor(pts_cur_, inliers_);
//        reduceVecor(fts_ref_, inliers_);
//        inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));
//    }
    //! get undistorted points
    std::vector<cv::Point2f> temp_udist;
    cv::undistortPoints(pts_cur_, temp_udist, K_, D_);
    fts_cur_.resize(pts_cur_.size());
    int idx = 0;
    std::for_each(temp_udist.begin(), temp_udist.end(), [&](cv::Point2f& pt){fts_cur_[idx].x = pt.x; fts_cur_[idx].y = pt.y; idx++;});

    SSVO_INFO_STREAM("[INIT] Avage disparity: " << disparity);
    if(disparity < Config::initMinDisparity()) return FAILURE;

    double t3 = (double)cv::getTickCount();
    //! [3] geometry check by F matrix
    //cv::Mat FE = cv::findEssentialMat(fts_ref_, fts_cur_, 1, cv::Point2d(0, 0), cv::RANSAC, 0.95, 0.001, inliers_);
    //int inliers_count = cv::countNonZero(inliers_);
    cv::Mat E;
    int inliers_count = Fundamental::findFundamentalMat(fts_ref_, fts_cur_, E, inliers_, Config::initUnSigma2(), Config::initMaxRansacIters(), true);
//    if(inliers_count!= pts_ref_.size())
//    {
//        reduceVecor(pts_ref_, inliers_);
//        reduceVecor(pts_cur_, inliers_);
//        reduceVecor(fts_ref_, inliers_);
//        reduceVecor(fts_cur_, inliers_);
//        inliers_ = cv::Mat(pts_ref_.size(), 1, CV_8UC1, cv::Scalar(255));
//    }
    SSVO_INFO_STREAM("[INIT] Inliers after epipolar geometry check: " << inliers_count);
    if(inliers_count < Config::initMinInliers()) return FAILURE;

    double t4 = (double)cv::getTickCount();
    //! [4] cheirality check
    cv::Mat R1, R2, t;
    Fundamental::decomposeEssentialMat(E, R1, R2, t);

    cv::Mat T, P3Ds;
    cv::Mat K =cv::Mat::eye(3,3,CV_64FC1);
    bool succeed = findBestRT(R1, R2, t, K, K, fts_ref_, fts_cur_, inliers_, P3Ds, T);
    if(!succeed) return FAILURE;
    SSVO_INFO_STREAM("[INIT] Inliers after cheirality check: " << cv::countNonZero(inliers_));

    double t5 = (double)cv::getTickCount();
    //! [5] reprojective check
    inliers_count = checkReprejectErr(pts_ref_, pts_cur_, fts_ref_, fts_cur_, T, inliers_, P3Ds, Config::initUnSigma2()*4, p3ds_);
    SSVO_INFO_STREAM("[INIT] Inliers after reprojective check: " << inliers_count);
    if(inliers_count < Config::initMinInliers()) return FAILURE;

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

void Initializer::getResults(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur,
    std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur, std::vector<Vector3d>& p3ds) const
{
    const int N = pts_ref_.size();
    pts_ref.reserve(N);
    pts_cur.reserve(N);
    fts_ref.reserve(N);
    fts_cur.reserve(N);

    const uchar* inliers_ptr = inliers_.ptr<uchar>(0);
    for (int i = 0; i < N; ++i) {
        if(!inliers_ptr[i])
            continue;

        pts_ref.push_back(pts_ref_[i]);
        pts_cur.push_back(pts_cur_[i]);
        fts_ref.push_back(fts_ref_[i]);
        fts_cur.push_back(fts_cur_[i]);
        p3ds.push_back(p3ds_[i]);
    }
}

void Initializer::getUndistInilers(std::vector<cv::Point2d>& fts_ref, std::vector<cv::Point2d>& fts_cur) const
{
    if(!finished_)
    {
        SSVO_WARN_STREAM("[INIT] Please waiting until inintialization finished!");
        return;
    }

    const int N = pts_ref_.size();
    fts_ref.reserve(N);
    fts_cur.reserve(N);

    const uchar* inliers_ptr = inliers_.ptr<uchar>(0);
    for (int i = 0; i < N; ++i) {
        if(!inliers_ptr[i])
            continue;

        fts_ref.push_back(fts_ref_[i]);
        fts_cur.push_back(fts_cur_[i]);
    }
}

void Initializer::getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const
{
    const int N = pts_ref_.size();
    pts_ref.reserve(N);
    pts_cur.reserve(N);

    const uchar* inliers_ptr = inliers_.ptr<uchar>(0);
    for (int i = 0; i < N; ++i) {
        if(!inliers_ptr[i])
            continue;

        pts_ref.push_back(pts_ref_[i]);
        pts_cur.push_back(pts_cur_[i]);
    }
}

void Initializer::kltTrack(const cv::Mat& img_ref, const cv::Mat& img_cur, const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur, cv::Mat& inliers)
{
    const int N = pts_ref.size();
    const int klt_win_size = 21;
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    const int border = 8;
    const int x_min = border;
    const int y_min = border;
    const int x_max = img_ref.cols - border;
    const int y_max = img_ref.rows - border;

    std::vector<cv::Point2f> pts_ref_temp;
    std::vector<cv::Point2f> pts_cur_temp;
    pts_ref_temp.reserve(N);
    pts_cur_temp.reserve(N);
    if(!inliers.empty())
    {
        const uchar* inliers_ptr = inliers.ptr<uchar>(0);
        int i = 0;
        for (int idx = 0; idx < N; ++idx)
        {
            if(!inliers_ptr[idx])
                continue;

            pts_ref_temp.push_back(pts_ref[idx]);
            pts_cur_temp.push_back(pts_cur[idx]);

        }
    }
    else
    {
        inliers = cv::Mat(N, 1, CV_8UC1, cv::Scalar(255));
        pts_ref_temp = pts_ref;
        pts_cur_temp = pts_cur;
    }


    std::vector<float> error;
    std::vector<uchar> status;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);

    cv::calcOpticalFlowPyrLK(
        img_ref, img_cur,
        pts_ref_temp, pts_cur_temp,
        status, error,
        cv::Size(klt_win_size, klt_win_size), 3,
        termcrit, cv::OPTFLOW_USE_INITIAL_FLOW
    );


    pts_cur.resize(N);
    uchar* inliers_ptr = inliers.ptr<uchar>(0);
    int idx = 0;
    for(int i = 0; i < N; ++i)
    {
        if(!inliers_ptr[i])
            continue;

        const cv::Point2f pt = pts_cur_temp[idx];
        if(!status[idx] || pt.x < x_min || pt.y < y_min || pt.x > x_max || pt.y > y_max)
        {
            inliers_ptr[i] = 0;
            pts_cur[i] = cv::Point2f(0,0);
        }
        else
        {
            pts_cur[i] = pt;
        }
        idx++;
    }
}

void Initializer::calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Mat& inliers, std::vector<std::pair<int, float> >& disparities)
{
    const int N = pts1.size();

    disparities.clear();
    disparities.reserve(N);
    const uchar* inliers_ptr = inliers.ptr<uchar>(0);
    for(int i = 0; i < N; i++)
    {
        if(!inliers_ptr[i])
            continue;

        float dx = pts2[i].x - pts1[i].x;
        float dy = pts2[i].y - pts1[i].y;
        disparities.push_back(std::make_pair(i,sqrt(dx*dx + dy*dy)));
    }
}

bool Initializer::findBestRT(const cv::Mat& R1, const cv::Mat& R2, const cv::Mat& t, const cv::Mat& K1, const cv::Mat& K2,
                             const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2, cv::Mat& mask, cv::Mat& P3Ds, cv::Mat& T)
{
    assert(fts1.size() == fts2.size());
    if(mask.empty())
        mask = cv::Mat(fts1.size(), 1, CV_8UC1, cv::Scalar(255));

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
    const double max_dist = 50.0;
    cv::Mat P3Ds1;
    triangulate(P0, P1, fts1, fts2, mask, P3Ds1);
    cv::Mat mask1 = mask.t();
    mask1 &= (P3Ds1.row(2) > 0) & (P3Ds1.row(2) < max_dist);
    T_P3Ds = T1*P3Ds1;
    mask1 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds2;
    triangulate(P0, P2, fts1, fts2, mask, P3Ds2);
    cv::Mat mask2 = mask.t();
    mask2 &= (P3Ds2.row(2) > 0) & (P3Ds2.row(2) < max_dist);
    T_P3Ds = T2*P3Ds2;
    mask2 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds3;
    triangulate(P0, P3, fts1, fts2, mask, P3Ds3);
    cv::Mat mask3 = mask.t();
    mask3 &= (P3Ds3.row(2) > 0) & (P3Ds3.row(2) < max_dist);
    T_P3Ds = T3*P3Ds3;
    mask3 &= (T_P3Ds.row(2) > 0) & (T_P3Ds.row(2) < max_dist);

    cv::Mat P3Ds4;
    triangulate(P0, P4, fts1, fts2, mask, P3Ds4);
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

int Initializer::checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur, const  std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                       const cv::Mat& T, cv::Mat& mask, const cv::Mat& P3Ds, const double sigma2, std::vector<Vector3d>& p3ds)
{
    assert(T.type() == CV_64FC1);
    assert(P3Ds.type() == CV_64FC1);

    const int size = pts_ref_.size();
    p3ds.clear(); p3ds.resize(size);

//    std::vector<int> inliers;
//    inliers.reserve(size);
    int inliers_count = 0;
    uchar* mask_ptr = mask.ptr<uchar>(0);
    const double* P3Ds_ptr = P3Ds.ptr<double>(0);
    const int strick = P3Ds.cols;
    const int strick2 = strick*2;

    for(int i = 0; i < size; ++i)
    {
        if(!mask_ptr[i])
            continue;

        const double* p3d1 = &P3Ds_ptr[i];
        const double X = p3d1[0];
        const double Y = p3d1[strick];
        const double Z = p3d1[strick2];
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

        cv::Mat P1 = (cv::Mat_<double>(4,1) <<X, Y, Z, 1);
        cv::Mat P2 = T*P1;
        double* p3d2 = P2.ptr<double>(0);
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

//        inliers.push_back(i);
        inliers_count++;
        p3ds[i] = Vector3d(X, Y, Z);
    }

//    const int n_inliers = inliers.size();
//    if(n_inliers == size)
//        return inliers.size();

//    std::vector<cv::Point2f>::iterator pts_ref_inter = pts_ref.begin();
//    std::vector<cv::Point2f>::iterator pts_cur_inter = pts_cur.begin();
//    std::vector<cv::Point2d>::iterator fts_ref_inter = fts_ref.begin();
//    std::vector<cv::Point2d>::iterator fts_cur_inter = fts_cur.begin();
//    for(int j = 0; j < n_inliers; ++j)
//    {
//        const int id = inliers[j];
//
//        *pts_ref_inter = pts_ref[id];
//        *pts_cur_inter = pts_cur[id];
//        *fts_ref_inter = fts_ref[id];
//        *fts_cur_inter = fts_cur[id];
//
//        pts_ref_inter++;
//        pts_cur_inter++;
//        fts_ref_inter++;
//        fts_cur_inter++;
//    }
//    pts_ref.resize(n_inliers);
//    pts_cur.resize(n_inliers);
//    fts_ref.resize(n_inliers);
//    fts_cur.resize(n_inliers);

    return inliers_count;
}

#if 0
void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& mask, cv::Mat& P3D)
{
    const int n = pts1.size();
    assert(n == pts2.size());
    assert(mask.cols == 1 && mask.rows == n);
    uchar *mask_ptr = mask.ptr<uchar>(0);
    P3D = cv::Mat::zeros(4, n, CV_64FC1);
    for(int i = 0; i < n; ++i)
    {
        cv::Mat wP;

        if(mask_ptr[i])
            triangulate(P1, P2, pts1[i], pts2[i], wP);
        else
            wP = cv::Mat::zeros(4,1,CV_64FC1);
        wP.copyTo(P3D.col(i));
    }
}
#else
void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2, cv::Mat& mask, cv::Mat& P3D)
{
    const int n = fts1.size();
    assert(n == fts2.size());
    assert(mask.cols == 1 && mask.rows == n);
    uchar *mask_ptr = mask.ptr<uchar>(0);
    MatrixXd eP1, eP2;
    cv::cv2eigen(P1, eP1);
    cv::cv2eigen(P2, eP2);
    MatrixXd eP3D = MatrixXd::Zero(4, n);
    P3D = cv::Mat::zeros(4, n, CV_64FC1);
    for(int i = 0; i < n; ++i)
    {
        Vector4d wP = Vector4d::Zero();

        if(mask_ptr[i]) {
            triangulate(eP1, eP2, fts1[i], fts2[i], wP);
        }

        eP3D.col(i) = wP;

    }
    cv::eigen2cv(eP3D, P3D);
}
#endif

void Initializer::triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2d& ft1, const cv::Point2d& ft2, cv::Mat& P3D)
{
    cv::Mat A(4,4,CV_64F);

    A.row(0) = ft1.x*P1.row(2)-P1.row(0);
    A.row(1) = ft1.y*P1.row(2)-P1.row(1);
    A.row(2) = ft2.x*P2.row(2)-P2.row(0);
    A.row(3) = ft2.y*P2.row(2)-P2.row(1);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    P3D = Vt.row(3).t();
    P3D = P3D/P3D.at<double>(3);
}

void Initializer::triangulate(const MatrixXd& P1, const MatrixXd& P2, const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D)
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

void Initializer::reduceVecor(std::vector<cv::Point2d>& fts, const cv::Mat& inliers)
{
    assert(inliers.cols == 1 || inliers.rows == 1);
    assert(inliers.type() == CV_8UC1);
    int size = MAX(inliers.cols, inliers.rows);
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

int Fundamental::findFundamentalMat(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, cv::Mat &E,
                                    cv::Mat& inliers, double sigma2, int max_iterations, const bool bE)
{
    assert(fts_prev.size() == fts_next.size());

    return runRANSAC(fts_prev, fts_next, E, inliers, sigma2, max_iterations, bE);
}

#if 1
void Fundamental::run8point(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, cv::Mat& F, const bool bE)
{
    const int N = fts_prev.size();
    assert(N >= 8);

    std::vector<cv::Point2d> fts_prev_norm;
    std::vector<cv::Point2d> fts_next_norm;
    cv::Mat T1, T2;
    Normalize(fts_prev, fts_prev_norm, T1);
    Normalize(fts_next, fts_next_norm, T2);

    cv::Mat A(N, 9, CV_64F);
    for(int i = 0; i < N; ++i)
    {
        const double u1 = fts_prev_norm[i].x;
        const double v1 = fts_prev_norm[i].y;
        const double u2 = fts_next_norm[i].x;
        const double v2 = fts_next_norm[i].y;
        double* a = A.ptr<double>(i);

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

    cv::Mat U, W, Vt;

    cv::eigen(A.t()*A, W, Vt);

    cv::Mat Fpre = Vt.row(8).reshape(0, 3);

    cv::Mat F_unorm = T2.t()*Fpre*T1;

    cv::SVDecomp(F_unorm, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    if(bE)
    {
        const double s = 0.5*(W.at<double>(0)+W.at<double>(1));
        W.at<double>(0) = s;
        W.at<double>(1) = s;
    }
    W.at<double>(2) = 0;

    F = U*cv::Mat::diag(W)*Vt;

    double F22 = F.at<double>(2, 2);
    if(fabs(F22) > std::numeric_limits<double>::epsilon())
        F /= F22;
}
#else
void Fundamental::run8point(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, cv::Mat& F)
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

    JacobiSVD<MatrixXf> svd1(Ft.transpose(), ComputeFullV|ComputeFullU);
    MatrixXd V = svd1.matrixV();
    MatrixXd U = svd1.matrixU();
    Vector3d S = svd1.singularValues();
    S(2) = 0;
    DiagonalMatrix<double, Dynamic> W(S);

    Matrix3d Fn = U * W * V.transpose();

    MatrixXd FF = T2.transpose()*Fn*T1;
    double F22 = FF(2, 2);
    if(fabs(F22) > std::numeric_limits<double>::epsilon())
        FF /= F22;

    cv::eigen2cv(FF, F);
}
#endif

int Fundamental::runRANSAC(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, cv::Mat& F, cv::Mat& inliers,
    const double sigma2, const int max_iterations, const bool bE)
{
    const int N = fts_prev.size();
    const int modelPoints = 8;

    const double threshold = 3.841*sigma2;
    const int max_iters = MIN(MAX(max_iterations, 1), 1000);

    std::vector<int> total_points;
    total_points.reserve(N);
    const uchar* inliers_ptr = inliers.ptr<uchar>(0);
    for(int i = 0; i < N; ++i)
    {
        if(inliers_ptr[i])
            total_points.push_back(i);
    }
    assert(total_points.size() >= modelPoints);

    std::vector<cv::Point2d> fts1(modelPoints);
    std::vector<cv::Point2d> fts2(modelPoints);
    std::vector<cv::Point2d> fts1_norm;
    std::vector<cv::Point2d> fts2_norm;
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
            fts1[i] = fts_prev[points[randi]];
            fts2[i] = fts_next[points[randi]];

            points[randi] = points.back();
            points.pop_back();
        }

        run8point(fts1, fts2, F_temp, bE);

        int inliers_count = 0;
        cv::Mat inliers_temp = cv::Mat::zeros(N, 1, CV_8UC1);
        uchar* inliers_ptr = inliers_temp.ptr<uchar>(0);
        std::vector<int>::iterator total_points_iter = total_points.begin();
        for(const std::vector<int>::iterator end_iter = total_points.end(); total_points_iter != end_iter; ++total_points_iter)
        {
            const int n = *total_points_iter;
            double error1, error2;
            computeErrors(fts_prev[n], fts_next[n], F_temp.ptr<double>(0), error1, error2);

            const double error = MAX(error1, error2);

            if(error < threshold)
            {
                inliers_ptr[n] = 0xff;
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

    fts1.clear();
    fts2.clear();
    for(int n = 0; n < N; ++n)
    {
        if(0 == inliers.at<uchar>(n, 0))
        {
            continue;
        }

        fts1.push_back(fts_prev[n]);
        fts2.push_back(fts_next[n]);
    }

    run8point(fts1, fts2, F, bE);

    return max_inliers;
}

void Fundamental::Normalize(const std::vector<cv::Point2d>& fts, std::vector<cv::Point2d>& fts_norm, cv::Mat& T)
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

    T = cv::Mat::eye(3,3,CV_64F);
    T.at<double>(0,0) = scale_x;
    T.at<double>(1,1) = scale_y;
    T.at<double>(0,2) = -mean.x*scale_x;
    T.at<double>(1,2) = -mean.y*scale_y;
}

void Fundamental::Normalize(const std::vector<cv::Point2d>& fts, std::vector<cv::Point2d>& fts_norm, Matrix3f& T)
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

inline void Fundamental::computeErrors(const cv::Point2d& p1, const cv::Point2d& p2, const double* F, double& err1, double& err2)
{
    //! point X1 = (u1, v1, 1)^T in first image
    //! poInt X2 = (u2, v2, 1)^T in second image
    const double u1 = p1.x;
    const double v1 = p1.y;
    const double u2 = p2.x;
    const double v2 = p2.y;

    //! epipolar line in the second image L2 = (a2, b2, c2)^T = F   * X1
    const double a2 = F[0]*u1 + F[1]*v1 + F[2];
    const double b2 = F[3]*u1 + F[4]*v1 + F[5];
    const double c2 = F[6]*u1 + F[7]*v1 + F[8];
    //! epipolar line in the first image  L1 = (a1, b1, c1)^T = F^T * X2
    const double a1 = F[0]*u2 + F[3]*v2 + F[6];
    const double b1 = F[1]*u2 + F[4]*v2 + F[7];
    const double c1 = F[2]*u2 + F[5]*v2 + F[8];

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

//! modified from OpenCV
void Fundamental::decomposeEssentialMat(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t)
{
    assert(E.cols == 3 && E.rows == 3);

    cv::Mat U, D,Vt;
    cv::SVD::compute(E, D, U, Vt);

    if(determinant(U) < 0) U *= -1.;
    if(determinant(Vt) < 0) Vt *= -1.;

    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;

    U.col(2).copyTo(t);
    t = t / cv::norm(t);
}

}