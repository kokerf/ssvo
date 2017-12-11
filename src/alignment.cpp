#include "utils.hpp"
#include "alignment.hpp"
#include "optimizer.hpp"


namespace ssvo{

//
// Align SE3
//

AlignSE3::AlignSE3(const int max_iterations, const double epslion):
    top_level_(Config::alignTopLevel()), //patch_area_(patch_size_*patch_size_),half_patch_size_(patch_size_/2),
    max_iterations_(max_iterations),
    epslion_squared_(epslion*epslion)
{

}

bool AlignSE3::run(Frame::Ptr reference_frame, Frame::Ptr current_frame)
{
    ref_frame_ = reference_frame;
    cur_frame_ = current_frame;
    std::vector<Feature::Ptr> fts = ref_frame_->getFeatures();
    const size_t N = fts.size();
    LOG_ASSERT(N != 0) << " AlignSE3: Frame(" << reference_frame->id_ << ") "<< " no features to track!" ;

    ref_feature_cache_.resize(NoChange, N);
    ref_patch_cache_.resize(N, NoChange);
    jacbian_cache_.resize(N*PatchArea, NoChange);

    T_cur_from_ref_ = current_frame->pose() * reference_frame->pose_inverse();
    LOG(INFO) << "T_cur_from_ref_ " << T_cur_from_ref_.log().transpose();

    for(int l = top_level_; l >= 0; l--)
    {
        const int n = computeReferencePatches(l);

        double res_old = std::numeric_limits<double>::max();
        Sophus::SE3d T_old = T_cur_from_ref_;
        for(int i = 0; i < max_iterations_; ++i)
        {
            //! compute residual
            double res = computeResidual(l, n);

            if(res > res_old)
            {
                T_cur_from_ref_ = T_old;
                break;
            }
            //! update
            res_old = res;
            T_old = T_cur_from_ref_;
            Sophus::SE3d::Tangent se3 = Hessian_.ldlt().solve(Jres_);
            T_cur_from_ref_ = T_cur_from_ref_ * Sophus::SE3d::exp(-se3);

            LOG(INFO) << "Level: " << l << " Residual: [" << res << "] update: [" << T_cur_from_ref_.log().transpose() << "]";

            //! termination
            if(se3.dot(se3) < epslion_squared_)
                 break;
        }
    }

    current_frame->setPose(T_cur_from_ref_ * reference_frame->pose());
    LOG(INFO) << "pose:\n " << T_cur_from_ref_.matrix3x4();

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
        Vector3d ref_xyz = fts[n]->ft;
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

//    cv::Mat res_img = cv::Mat::zeros(rows, cols, CV_64FC1);
    cv::Mat showimg = cv::Mat::zeros(rows, cols, CV_8UC1);
    for(int n = 0; n < N; ++n)
    {
        const Vector3d cur_xyz = T_cur_from_ref_ * ref_feature_cache_.col(n);
        const Vector2d cur_px = cur_frame_->cam_->project(cur_xyz) * scale;
        if(cur_px[0] < border || cur_px[1] < border || cur_px[0] + border > cols - 1 || cur_px[1] + border > rows - 1)
            continue;

//        LOG(INFO) << cur_xyz.transpose() << " " << cur_px.transpose();
        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(cur_eigen_img, residual, cur_px[0], cur_px[1]);
        residual.noalias() -= ref_patch_cache_.row(n);//.transpose();
        Matrix<double, PatchArea, 6, RowMajor> J = jacbian_cache_.block(n*PatchArea, 0, PatchArea, 6);
        //Matrix<double, PatchArea, 6, RowMajor> J1 = Eigen::Map<Matrix<double, PatchArea, 6, RowMajor> >(jacbian_cache_.data()+n*PatchArea*6);

        Jres_.noalias() -= J.transpose() * residual;
        Hessian_.noalias() += J.transpose() * J;

        res += residual.dot(residual)/PatchArea;
        count++;

        cv::Mat mat(PatchSize, PatchSize, CV_64FC1, residual.data());
        cv::Mat mat_temp;
        Vector2i start = cur_px.cast<int>() - Vector2i(HalfPatchSize,HalfPatchSize);
        Vector2i end = start + Vector2i(PatchSize,PatchSize);

        mat.copyTo(mat_temp);
        mat_temp = cv::abs(mat_temp);
        mat_temp.convertTo(mat_temp, CV_8UC1);
        mat_temp.copyTo(showimg.rowRange(start[1], end[1]).colRange(start[0], end[0]));

    }

    cv::imshow("res", showimg);
    cv::waitKey(0);

    return res/count;
}

//
// Align Patch
//
bool Align2DI::run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
                   const Matrix<double, PatchArea, 1> &patch,
                   const Matrix<double, PatchArea, 1> &patch_gx,
                   const Matrix<double, PatchArea, 1> &patch_gy,
                   Eigen::Vector3d &estimate,
                   const int MAX_ITER,
                   const double EPS)
{
    const double min_update_squared = EPS*EPS;
    bool converged = false;
    estimate_ = estimate;

    //! Per-compute
    jacbian_cache_.resize(PatchArea, 3);
    Jres_.resize(3);

    //! get jacobian
    jacbian_cache_.col(0) = patch_gx;
    jacbian_cache_.col(1) = patch_gy;
    jacbian_cache_.col(2).setConstant(1);

    Hessian_ = jacbian_cache_.transpose() * jacbian_cache_;
    invHessian_ = Hessian_.inverse();

    Eigen::Vector3d update;
    double res_old = std::numeric_limits<double>::max();
    for(int iter = 0; iter < MAX_ITER; iter++)
    {
        const double u = estimate_[0];
        const double v = estimate_[1];
        const double idiff = estimate_[2];

        if(u < border_ || v < border_ || u + border_ >= image.cols() - 1 || v + border_ >= image.rows() - 1)
        {
            std::cerr << "Error! The estimate pixel location is out of the scope!" << std::endl;
            return false;
        }

        // compute interpolation weights
        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(image, residual, u, v);
        residual.noalias() -= patch;
        residual.array() += idiff;

//        double res = residual.norm();
//        if(res > res_old)
//        {
//            estimate_.noalias() += update;
//            break;
//        }
//        res_old = res;

        Jres_ = jacbian_cache_.transpose() * residual;

        //! update
        update = invHessian_ * Jres_;
        estimate_.noalias() -= update;

//        LOG(INFO) << "iter:" << iter << " estimate: [" << estimate_[0] << ", " << estimate_[1] << ", " << estimate_[2]
//                  << "] update: [" << std::setw(12) << update.transpose() << "] res: " << residual.norm()/PatchArea;

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    estimate = estimate_;
    return converged;
}


}