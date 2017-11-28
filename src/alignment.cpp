#include "utils.hpp"
#include "alignment.hpp"


namespace ssvo{

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
    ref_patch_cache_.resize(N, PatchArea);
    jacbian_cache_.resize(N*PatchArea, NoChange);

    T_cur_from_ref_ = current_frame->pose() * reference_frame->pose_inverse();

    for(int l = top_level_; l >= 0; l--)
    {
        const int n = computeReferencePatches(l);

        for (int i = 0; i < max_iterations_; ++i) {
            double res = computeResidual(l, n);

            Sophus::SE3d::Tangent se3 = Hessian_.ldlt().solve(Jres_);

            LOG(INFO) << "Residual: " << res << " update: " << se3.transpose();

            if(se3.dot(se3) < epslion_squared_)
                break;

            T_cur_from_ref_ = T_cur_from_ref_ * Sophus::SE3d::exp(-se3);
        }
    }
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
    Matrix<uchar , Dynamic, Dynamic, RowMajor> eigen_img = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)ref_img.data, rows, cols);
    const int stride = ref_img.cols;
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
        std::cout << "ref_xyz " << ref_xyz.transpose() << std::endl;
        ref_xyz.normalize();
        std::cout << "ref_xyz " << ref_xyz.transpose() << std::endl;
        ref_xyz *= depth;

        Vector3d xyz = ref_frame_->pose() * fts[n]->mpt->pose();

        std::cout <<" xyz \n" << ref_xyz.transpose() << " \n" << xyz.transpose() << std::endl;
        ref_feature_cache_.col(feature_counter) = ref_xyz;

        //! compute jacbian(with -)
        Matrix<double, 2, 6, RowMajor> J;
        Frame::jacobian_xyz2uv(ref_xyz, J);

        Matrix<double, PatchArea, 1> img, dx, dy;
        utils::interpolateMat<uchar, double, PatchSize>(eigen_img, img, dx, dy, ref_px[0], ref_px[1]);
        ref_patch_cache_.row(feature_counter) = img;
        //LOG(INFO) << "J:" << fx * dx * J.row(0) + fy * dy * J.row(1);
        jacbian_cache_.block(feature_counter*PatchArea, 0, PatchArea, 6) = fx * dx * J.row(0) + fy * dy * J.row(1);

        //! visiable feature counter
        feature_counter++;
    }
    //LOG(INFO) << "JAC:\n" << jacbian_cache_;

    return feature_counter;
}

double AlignSE3::computeResidual(int level, int N)
{
    const cv::Mat cur_img = cur_frame_->getImage(level);
    const double scale = 1.0f/(1<<level);
    const int cols = cur_img.cols;
    const int rows = cur_img.rows;
    const int border = HalfPatchSize+1;
    Matrix<uchar , Dynamic, Dynamic, RowMajor> eigen_img = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)cur_img.data, rows, cols);
    Hessian_.setZero();
    Jres_.setZero();
    double res = 0;
    int count = 0;

    cv::Mat res_img = cv::Mat::zeros(rows, cols, CV_64FC1);
    for(int n = 0; n < N; ++n)
    {
        const Vector3d cur_xyz = T_cur_from_ref_ * ref_feature_cache_.col(n);
        const Vector2d cur_px = cur_frame_->cam_->project(cur_xyz) * scale;
        if(cur_px[0] < border || cur_px[1] < border || cur_px[0] + border > cols - 1 || cur_px[1] + border > rows - 1)
            continue;

        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(eigen_img, residual, cur_px[0], cur_px[1]);
        residual.noalias() -= ref_patch_cache_.row(n);//.transpose();
        Matrix<double, PatchArea, 6, RowMajor> J = jacbian_cache_.block(n*PatchArea, 0, PatchArea, 6);
        //Matrix<double, PatchArea, 6, RowMajor> J1 = Eigen::Map<Matrix<double, PatchArea, 6, RowMajor> >(jacbian_cache_.data()+n*PatchArea*6);

        Jres_.noalias() -= J.transpose() * residual;
        Hessian_.noalias() += J.transpose() * J;

        res = residual.dot(residual);
        count++;

        for (int i = 0; i < PatchSize; ++i) {

            Matrix<double, PatchSize, 1> img_residual = Eigen::Map<Matrix<double, PatchSize, 1> >(res_img.ptr<double>(cur_px[1]-HalfPatchSize+i)+((int)cur_px[0])-HalfPatchSize);
            img_residual = residual.segment(i*PatchSize, PatchSize);
        }
    }
    cv::imshow("res", res_img);
    cv::waitKey(0);

    return res/count;
}

}