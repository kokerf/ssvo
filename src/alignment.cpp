#include "utils.hpp"
#include "alignment.hpp"
#include "optimizer.hpp"


namespace ssvo{

//
// Align SE3
//

AlignSE3::AlignSE3(bool verbose, bool visible):
    verbose_(verbose), visible_(visible)
{}

bool AlignSE3::run(Frame::Ptr reference_frame,
                   Frame::Ptr current_frame,
                   int top_level,
                   int max_iterations,
                   double epslion)
{
    double epslion_squared = epslion*epslion;
    ref_frame_ = reference_frame;
    cur_frame_ = current_frame;

    const size_t N = ref_frame_->features().size();
    LOG_ASSERT(N != 0) << " AlignSE3: Frame(" << reference_frame->id_ << ") "<< " no features to track!" ;

    ref_feature_cache_.resize(NoChange, N);
    ref_patch_cache_.resize(N, NoChange);
    jacbian_cache_.resize(N*PatchArea, NoChange);

    T_cur_from_ref_ = cur_frame_->Tcw() * ref_frame_->pose();
    LOG_IF(INFO, verbose_) << "T_cur_from_ref_ " << T_cur_from_ref_.log().transpose();

    for(int l = top_level; l >= 0; l--)
    {
        const int n = computeReferencePatches(l);

        double res_old = std::numeric_limits<double>::max();
        SE3d T_cur_from_ref_old = T_cur_from_ref_;
        for(int i = 0; i < max_iterations; ++i)
        {
            //! compute residual
            double res = computeResidual(l, n);

            if(res > res_old)
            {
                T_cur_from_ref_ = T_cur_from_ref_old;
                break;
            }
            //! update
            res_old = res;
            T_cur_from_ref_old = T_cur_from_ref_;
            SE3d::Tangent se3 = Hessian_.ldlt().solve(Jres_);
            T_cur_from_ref_ = T_cur_from_ref_ * SE3d::exp(-se3);

            LOG_IF(INFO, verbose_) << "Level: " << l << " Residual: " << res << " step: " << se3.dot(se3)
                                   << " SE3: [" << T_cur_from_ref_.log().transpose() << "]";

            //! termination
            if(se3.dot(se3) < epslion_squared)
                 break;
        }
    }

    cur_frame_->setTcw(T_cur_from_ref_*ref_frame_->Tcw());
    LOG_IF(INFO, verbose_) << "T_cur_from_ref:\n " << T_cur_from_ref_.matrix3x4();

    return true;
}

int AlignSE3::computeReferencePatches(int level)
{
    std::vector<Feature::Ptr> fts = ref_frame_->getFeatures();
    const size_t N = fts.size();

    Vector3d ref_pose = ref_frame_->pose().translation();
    const cv::Mat ref_img = ref_frame_->getImage(level);
    const int cols = ref_img.cols;
    const int rows = ref_img.rows;
    const int border = HalfPatchSize+1;
    Matrix<uchar , Dynamic, Dynamic, RowMajor> ref_eigen_img = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)ref_img.data, rows, cols);
    const double scale = 1.0f/(1<<level);
    const double fx = ref_frame_->cam_->fx() * scale;
    const double fy = ref_frame_->cam_->fy() * scale;

    int feature_counter = 0;
    for(size_t n = 0; n < N; ++n)
    {
        Vector2d ref_px = fts[n]->px*scale;
        if(fts[n]->mpt == nullptr ||
            ref_px[0] < border || ref_px[1] < border || ref_px[0] + border > cols - 1 || ref_px[1] + border > rows - 1)
            continue;

        double depth = (fts[n]->mpt->pose() - ref_pose).norm();
        Vector3d ref_xyz = fts[n]->fn;
        ref_xyz *= depth;

        ref_feature_cache_.col(feature_counter) = ref_xyz;

        //! compute jacbian(with -)
        Matrix<double, 2, 6, RowMajor> J;
        Frame::jacobian_xyz2uv(ref_xyz, J);

        Matrix<double, PatchArea, 1> img, dx, dy;
        utils::interpolateMat<uchar, double, PatchSize>(ref_eigen_img, img, dx, dy, ref_px[0], ref_px[1]);
        ref_patch_cache_.row(feature_counter) = img;
        jacbian_cache_.block(feature_counter*PatchArea, 0, PatchArea, 6) = fx * dx * J.row(0) + fy * dy * J.row(1);

        //! visiable feature counter
        feature_counter++;
    }

    return feature_counter;
}

double AlignSE3::computeResidual(int level, int N)
{
    const cv::Mat cur_img = cur_frame_->getImage(level);
    const double scale = 1.0f/(1<<level);
    const int cols = cur_img.cols;
    const int rows = cur_img.rows;
    const int border = HalfPatchSize+1;
    Matrix<uchar , Dynamic, Dynamic, RowMajor> cur_eigen_img = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)cur_img.data, rows, cols);
    Hessian_.setZero();
    Jres_.setZero();
    double res = 0;
    int count = 0;

    cv::Mat showimg = cv::Mat::zeros(rows, cols, CV_8UC1);
    for(int n = 0; n < N; ++n)
    {
        const Vector3d cur_xyz = T_cur_from_ref_ * ref_feature_cache_.col(n);
        const Vector2d cur_px = cur_frame_->cam_->project(cur_xyz) * scale;
        if(cur_px[0] < border || cur_px[1] < border || cur_px[0] + border > cols - 1 || cur_px[1] + border > rows - 1)
            continue;

        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(cur_eigen_img, residual, cur_px[0], cur_px[1]);
        residual.noalias() -= ref_patch_cache_.row(n);
        Matrix<double, PatchArea, 6, RowMajor> J = jacbian_cache_.block(n*PatchArea, 0, PatchArea, 6);

        Jres_.noalias() -= J.transpose() * residual;
        Hessian_.noalias() += J.transpose() * J;

        res += residual.dot(residual)/PatchArea;
        count++;

        if(visible_)
        {
            cv::Mat mat_double(PatchSize, PatchSize, CV_64FC1, residual.data());
            mat_double = cv::abs(mat_double);
            Vector2i start = cur_px.cast<int>() - Vector2i(HalfPatchSize,HalfPatchSize);
            Vector2i end = start + Vector2i(PatchSize,PatchSize);
            cv::Mat mat_uchar;
            mat_double.convertTo(mat_uchar, CV_8UC1);
            mat_uchar.copyTo(showimg.rowRange(start[1], end[1]).colRange(start[0], end[0]));
        }

    }

    if(visible_)
    {
        cv::imshow("res", showimg);
        cv::waitKey(0);
    }

    return res/count;
}

//
// Align Patch
//
bool Align2DI::run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
                   const Matrix<double, PatchArea, 1> &patch,
                   const Matrix<double, PatchArea, 1> &patch_gx,
                   const Matrix<double, PatchArea, 1> &patch_gy,
                   Vector3d &estimate,
                   const int max_iterations,
                   const double epslion)
{
    const double min_update_squared = epslion*epslion;
    bool converged = false;
    estimate_ = estimate;

    //! Per-compute
    jacbian_cache_.resize(PatchArea, NoChange);

    //! get jacobian
    jacbian_cache_.col(0) = patch_gx;
    jacbian_cache_.col(1) = patch_gy;
    jacbian_cache_.col(2).setConstant(1);

    Hessian_ = jacbian_cache_.transpose() * jacbian_cache_;
    if(Hessian_.determinant() < 1e-10)
        return false;

    invHessian_ = Hessian_.inverse();

    Vector3d update;

    for(int iter = 0; iter < max_iterations; iter++)
    {
        const double u = estimate_[0];
        const double v = estimate_[1];
        const double idiff = estimate_[2];

        if(u < border_ || v < border_ || u + border_ >= image.cols() - 1 || v + border_ >= image.rows() - 1)
        {
            LOG_IF(INFO, verbose_) << "WARNING! The estimate pixel location is out of the scope!";
            return false;
        }

        // compute interpolation weights
        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(image, residual, u, v);
        residual.noalias() -= patch;
        residual.array() += idiff;

        Jres_ = jacbian_cache_.transpose() * residual;

        //! update
        update = invHessian_ * Jres_;
        estimate_.noalias() -= update;

        using std::to_string;
        std::string log = "iter:" + to_string(iter) + " res: " + to_string(residual.norm()/PatchArea) +
                          " estimate: [" + to_string(estimate_[0]) + ", " + to_string(estimate_[1]) + ", " + to_string(estimate_[2]) + "]\n";
        logs_.push_back(log);

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    std::string output;
    std::for_each(logs_.begin(), logs_.end(), [&](const std::string &s){output += s;});
    LOG_IF(INFO, verbose_) << output;

    estimate = estimate_;
    return converged;
}


namespace utils{

int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level)
{
    // Compute patch level in other image
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while(D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

void getWarpMatrixAffine(const Camera::Ptr &cam_ref,
                         const Camera::Ptr &cam_cur,
                         const Vector2d &px_ref,
                         const Vector3d &f_ref,
                         const int level_ref,
                         const double depth_ref,
                         const SE3d &T_cur_ref,
                         const int patch_size,
                         Matrix2d &A_cur_ref)
{
    const double half_patch_size = static_cast<double>(patch_size+2)/2;
    const Vector3d xyz_ref(depth_ref * f_ref);
    const double length = half_patch_size * (1 << level_ref);
    Vector3d xyz_ref_du(cam_ref->lift(px_ref + Vector2d(length, 0)));
    Vector3d xyz_ref_dv(cam_ref->lift(px_ref + Vector2d(0, length)));
    xyz_ref_du *= xyz_ref[2]/xyz_ref_du[2];
    xyz_ref_dv *= xyz_ref[2]/xyz_ref_dv[2];
    const Vector2d px_cur(cam_cur->project(T_cur_ref * xyz_ref));
    const Vector2d px_du(cam_cur->project(T_cur_ref * xyz_ref_du));
    const Vector2d px_dv(cam_cur->project(T_cur_ref * xyz_ref_dv));
    A_cur_ref.col(0) = (px_du - px_cur)/half_patch_size;
    A_cur_ref.col(1) = (px_dv - px_cur)/half_patch_size;
}

}

}