#include <opencv2/opencv.hpp>
#include "utils.hpp"

namespace ssvo {

namespace utils {

void kltTrack(const ImgPyr &imgs_ref, const ImgPyr &imgs_cur, const cv::Size win_size,
              const std::vector<cv::Point2f> &pts_ref, std::vector<cv::Point2f> &pts_cur,
              std::vector<bool> &status, cv::TermCriteria termcrit, bool track_forward, bool verbose)
{
    const size_t total_size = pts_ref.size();
    const int border = 8;
    const int x_min = border;
    const int y_min = border;
    const int x_max = imgs_ref[0].cols - border;
    const int y_max = imgs_cur[0].rows - border;

    std::vector<cv::Point2f> pts_ref_to_track;
    std::vector<cv::Point2f> pts_cur_tracked;
    std::vector<int> inlier_ids;

    if(status.empty())
    {
        pts_ref_to_track = pts_ref;
        pts_cur_tracked = pts_cur;
        inlier_ids.resize(total_size);
        for(size_t i = 0; i < total_size; ++i)
            inlier_ids[i] = i;
    }
    else
    {
        assert(status.size() == total_size);
        pts_ref_to_track.reserve(total_size);
        pts_cur_tracked.reserve(total_size);
        inlier_ids.reserve(total_size);
        for(size_t i = 0; i < total_size; ++i)
        {
            if(!status[i])
                continue;

            pts_ref_to_track.push_back(pts_ref[i]);
            pts_cur_tracked.push_back(pts_cur[i]);//! inital flow
            inlier_ids.push_back(i);
        }
    }

    const int track_size = inlier_ids.size();
    LOG_IF(INFO, verbose) << "Points for tracking: " << track_size;

    if(track_size <= 0)
        return;

    std::vector<float> error;
    std::vector<uchar> status_forward;

    //! forward track
    cv::calcOpticalFlowPyrLK(imgs_ref, imgs_cur, pts_ref_to_track, pts_cur_tracked, status_forward, error,
                             win_size, 3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    status.resize(total_size, false);
    std::fill(status.begin(), status.end(), false);
    for(int i = 0; i < track_size; ++i) {
        const cv::Point2f &pt = pts_cur_tracked[i];
        const int idx = inlier_ids[i];
        if(!status_forward[i] || pt.x < x_min || pt.y < y_min || pt.x > x_max || pt.y > y_max)
        {
            pts_cur[idx] = cv::Point2f(0, 0);
        }
        else
        {
            pts_cur[idx] = pt;
            status[idx] = true;
        }
    }

    LOG_IF(INFO, verbose) << "First tracked points: " << std::count(status_forward.begin(), status_forward.end(), true);
    if(!track_forward)
        return;

    std::vector<cv::Point2f> pts_cur_to_track;
    std::vector<cv::Point2f> pts_ref_tracked;
    inlier_ids.clear();
    inlier_ids.reserve(track_size);
    pts_cur_to_track.reserve(track_size);
    pts_ref_tracked.reserve(track_size);
    for(size_t i = 0; i < total_size; ++i)
    {
        if(!status[i])
            continue;

        pts_cur_to_track.push_back(pts_cur[i]);
        pts_ref_tracked.push_back(pts_cur[i]);
        inlier_ids.push_back(i);
    }

    if(inlier_ids.empty())
        return;

    //! backward track
    std::vector<uchar> status_back;
    cv::calcOpticalFlowPyrLK(imgs_cur, imgs_ref, pts_cur_to_track, pts_ref_tracked, status_back, error,
                             win_size, 3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    LOG_IF(INFO, verbose) << "Second tracked points: " << std::count(status_back.begin(), status_back.end(), true);

    int out = 0;
    const int N1 = inlier_ids.size();
    for(int i = 0; i < N1; ++i)
    {
        const int idx = inlier_ids[i];
        const cv::Point2f &pt_real = pts_ref[idx];
        const cv::Point2f &pt_estm = pts_ref_tracked[i];
        const cv::Point2f delta = pt_real - pt_estm;
        if(!status_back[i] || (delta.x * delta.x + delta.y * delta.y) > 2.0)
        {
            status[idx] = false;
            pts_cur[idx] = cv::Point2f(0, 0);
            out++;
        }
    }

    LOG_IF(INFO, verbose) << "Reject points: " << out << " final: " << std::count(status.begin(), status.end(), true);

}

bool Fundamental::findFundamentalMat(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d &F,
                                    std::vector<bool> &inliers, double sigma2, int max_iterations, const bool bE)
{
    assert(fts_prev.size() == fts_next.size());

    return runRANSAC(fts_prev, fts_next, F, inliers, sigma2, max_iterations, bE);
}

bool Fundamental::run8point(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d& F, const bool bE)
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
    if(isnan(F22))
        return false;

    F /= F22;
    return true;
}

bool Fundamental::runRANSAC(const std::vector<cv::Point2d>& fts_prev, const std::vector<cv::Point2d>& fts_next, Matrix3d& F,
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
    bool succeed = false;
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

        succeed = run8point(fts1, fts2, F_temp, bE);

        if(!succeed)
            continue;

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

    if(!succeed)
        return false;

    fts1.clear();
    fts2.clear();
    for(int n = 0; n < N; ++n)
    {
        if(!inliers[n])
            continue;

        fts1.push_back(fts_prev[n]);
        fts2.push_back(fts_next[n]);
    }

    Matrix3d F1;
    if(run8point(fts1, fts2, F1, bE))
        F = F1;

    return true;
}

bool triangulate(const Matrix3d& R_cr,  const Vector3d& t_cr, const Vector3d& fn_r, const Vector3d& fn_c, double &d_ref)
{
    Vector3d R_fn_r(R_cr * fn_r);
    Vector2d b(t_cr.dot(R_fn_r), t_cr.dot(fn_c));
    double A[4] = { R_fn_r.dot(R_fn_r), 0,
                    R_fn_r.dot(fn_c), -fn_c.dot(fn_c)};
    A[1] = -A[2];
    double det = A[0]*A[3] - A[1]*A[2];
    if(std::abs(det) < 0.000001)
        return false;

    d_ref = std::abs((b[0]*A[3] - A[1]*b[1])/det);
    return true;
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

double Fundamental::computeErrorSquared(const Vector3d &p1, const Vector3d &p2, const SE3d &T, const Vector2d &p)
{
    const Vector3d A(T*p1);
    const Vector3d B(T*p2);
    const Vector2d a = A.head<2>()/A[2];
    const Vector2d b = B.head<2>()/B[2];

    Vector2d ap = p - a;
    Vector2d ab = b - a;
    double ap_costh = ap.dot(ab) / ab.norm();

    return ap.squaredNorm() - ap_costh*ap_costh;
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
}