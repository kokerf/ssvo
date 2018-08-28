#include <iomanip>
#include "feature_detector.hpp"
#include "image_alignment.hpp"
#include "utils.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

using namespace ssvo;

//struct ResidualError {
//    ResidualError(const cv::Mat &image, const Matrix<double, 16, 1> &measured, const Vector3d &xyz, const Camera::Ptr &cam):
//        measured_(measured), xyz_(xyz), cam_(cam), cols_(image.cols), rows_(image.rows)
//    {
//        image_ = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)image.data, rows_, cols_);
//        border_x_[0] = 8;
//        border_x_[1] = cols_ -1 - 8;
//        border_y_[0] = 8;
//        border_y_[1] = rows_ -1 - 8;
//    }
//
//    template<typename T>
//    bool operator()(const T *const camera, T *residuals) const
//    {
//        //! In Sophus, stored in the form of [q, t]
//        const Sophus::SE3<T> pose = Eigen::Map<const Sophus::SE3<T> >(camera);
//        Eigen::Map<Matrix<T, 16, 1> > res(residuals);
//        Eigen::Matrix<T, 3, 1> xyz_in_cur = pose.rotationMatrix() * xyz_.cast<T>() + pose.translation() ;
//
//        const Eigen::Vector2d px = cam_->project(xyz_in_cur.template cast<double>());
//        if(px[0] < border_x_[0] || px[1] < border_y_[0] || px[0] > border_x_[1] || px[1] > border_y_[1])
//        {
//            res.setZero();
//            return false;
//        }
//
//        Eigen::Matrix<double, 16, 1> predicted, dx, dy;
//        utils::interpolateMat<uchar, double, 4>(image_, predicted, dx, dy, px[0], px[1]);
//
//        res = (predicted - measured_).cast<T>();//! I(w(x)) - T(x)
//        return true;
//    }
//
//    static ceres::CostFunction *Create(const cv::Mat &image, const Matrix<double, 16, 1> &measured, const Vector3d &xyz, const Camera::Ptr &cam)
//    {
//        return (new ceres::AutoDiffCostFunction<ResidualError, 16, Sophus::SE3d::num_parameters>(
//            new ResidualError(image, measured, xyz, cam)));
//    }
//
//    const Matrix<double, 16, 1> measured_;
//    const Vector3d xyz_;
//    const Camera::Ptr cam_;
//    const int cols_, rows_;
//    Vector2d border_x_;
//    Vector2d border_y_;
//
//    Matrix<uchar , Dynamic, Dynamic, RowMajor> image_;
//};

class ResidualErrorSE3 : public ceres::SizedCostFunction<16, 7>
{
public:

    ResidualErrorSE3(const cv::Mat &image, const Matrix<double, 16, 1> &measured, const Vector3d &xyz, const AbstractCamera::Ptr &cam):
        image_(image), measured_(measured), xyz_(xyz), cam_(cam), cols_(image.cols), rows_(image.rows)
    {
        border_x_[0] = 8;
        border_x_[1] = cols_ -1 - 8;
        border_y_[0] = 8;
        border_y_[1] = rows_ -1 - 8;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //! In Sophus, stored in the form of [q, t]
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);

        Eigen::Vector3d xyz_in_cur = q * xyz_ + t;

        const Eigen::Vector2d px = cam_->project(xyz_in_cur);
        if(px[0] < border_x_[0] || px[1] < border_y_[0] || px[0] > border_x_[1] || px[1] > border_y_[1])
        {
            Eigen::Map<Matrix<double, 16, 1> > res(residuals);
            res.setZero();//! I(w(x)) - T(x)

            if(!jacobians || !jacobians[0]) return true;
            Eigen::Map<Eigen::Matrix<double, 16, 7, Eigen::RowMajor> > jacobian(jacobians[0]);
            jacobian.setZero();

            return true;
        }

        Eigen::Matrix<double, 16, 1> predicted, dx, dy;
        utils::interpolateMat<uchar, double, 4>(image_, predicted, dx, dy, px[0], px[1]);

        Eigen::Map<Matrix<double, 16, 1> > res(residuals);
        res = predicted - measured_;//! I(w(x)) - T(x)

        if(!jacobians || !jacobians[0]) return true;
        Eigen::Map<Eigen::Matrix<double, 16, 7, Eigen::RowMajor> > jacobian(jacobians[0]);

        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jacobian_uv_ksai;
        const double x = xyz_in_cur[0];
        const double y = xyz_in_cur[1];
        const double z_inv = 1./xyz_in_cur[2];
        const double z_inv_2 = z_inv*z_inv;

        jacobian_uv_ksai(0,0) = z_inv;
        jacobian_uv_ksai(0,1) = 0.0;
        jacobian_uv_ksai(0,2) = -x*z_inv_2;
        jacobian_uv_ksai(0,3) = -y*jacobian_uv_ksai(0,2);
        jacobian_uv_ksai(0,4) = 1.0 - x*jacobian_uv_ksai(0,2);
        jacobian_uv_ksai(0,5) = -y*z_inv;

        jacobian_uv_ksai(1,0) = 0.0;
        jacobian_uv_ksai(1,1) = z_inv;
        jacobian_uv_ksai(1,2) = -y*z_inv_2;
        jacobian_uv_ksai(1,3) = 1.0 + y*jacobian_uv_ksai(1,2);
        jacobian_uv_ksai(1,4) = -jacobian_uv_ksai(0,3);
        jacobian_uv_ksai(1,5) = -x*z_inv;

        Eigen::Matrix<double, 16, 2, Eigen::RowMajor> jacobian_pixel_uv;
        jacobian_pixel_uv.col(0) = cam_->fx() * dx;
        jacobian_pixel_uv.col(1) = cam_->fy() * dy;

        jacobian.block<16, 6>(0,0) = jacobian_pixel_uv * jacobian_uv_ksai;
        jacobian.rightCols(1).setZero();

//        LOG(INFO) << "res:\n" << res.transpose();
//        LOG(INFO) << "jac:\n" << jacobian;

        return true;
    }

    static inline ceres::CostFunction *Create(const cv::Mat &image, const Matrix<double, 16, 1> &measured,
                                              const Vector3d &xyz, const AbstractCamera::Ptr &cam) {
        return (new ResidualErrorSE3(image, measured, xyz, cam));
    }

private:

    cv::Mat image_;

    const Matrix<double, 16, 1> measured_;
    const Vector3d xyz_;
    const AbstractCamera::Ptr cam_;
    const int cols_, rows_;
    Vector2d border_x_;
    Vector2d border_y_;

}; // class ResidualErrorSE3



Eigen::Matrix<double, 16, 1> ResidualError(const ceres::Problem& problem, ceres::ResidualBlockId id)
{
    auto cost = problem.GetCostFunctionForResidualBlock(id);
    std::vector<double*> parameterBlocks;
    problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
    Eigen::Matrix<double, 16, 1> residual;
    cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
    return residual;
}

void align_by_ceres(Frame::Ptr reference_frame, Frame::Ptr current_frame, int level)
{
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();

    reference_frame->optimal_Tcw_ = reference_frame->Tcw();
    current_frame->optimal_Tcw_ = current_frame->Tcw();

    problem.AddParameterBlock(reference_frame->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
    problem.AddParameterBlock(current_frame->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(reference_frame->optimal_Tcw_.data());

    const cv::Mat ref_img = reference_frame->getImage(level);
    const int cols = ref_img.cols;
    const int rows = ref_img.rows;
    Matrix<uchar , Dynamic, Dynamic, RowMajor> ref_eigen_img = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)ref_img.data, rows, cols);
    const double scale = 1.0f/(1<<level);
    const int border = 4+1;
    Vector3d ref_pose = reference_frame->pose().translation();

    std::vector<Feature::Ptr> fts = reference_frame->getFeatures();
    for(Feature::Ptr ft : fts)
    {
        Vector2d ref_px = ft->px_*scale;
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr ||
            ref_px[0] < border || ref_px[1] < border || ref_px[0] + border > cols - 1 || ref_px[1] + border > rows - 1)
            continue;

        double depth = (ft->mpt_->pose() - ref_pose).norm();
        Vector3d ref_xyz = ft->fn_;
        ref_xyz *= depth;

        Matrix<double, 16, 1> patch, dx, dy;
        utils::interpolateMat<uchar, double, 4>(ref_img, patch, ref_px[0], ref_px[1]);

        ceres::CostFunction* cost_function = ResidualErrorSE3::Create(current_frame->getImage(level), patch, mpt->pose(), current_frame->cam_);
        problem.AddResidualBlock(cost_function, NULL, current_frame->optimal_Tcw_.data());
    }

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solve(options, &problem, &summary);

    //! update pose
    current_frame->setPose(current_frame->optimal_Tcw_.inverse());

    LOG(INFO) << summary.FullReport();


    std::vector<ceres::ResidualBlockId> ids;
    problem.GetResidualBlocks(&ids);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        LOG(INFO) << "BlockId: " << std::setw(5) << i <<" residual(RMSE): " << ResidualError(problem, ids[i]).norm();
    }
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 5) << "Usge: ./test_alignment calib_file config_file path_to_sequence path_to_association";

    TUMDataReader dataset(argv[3], argv[4], true);

    std::string rgb_file0, rgb_file1, depth_file0, depth_file1;
    double timestamp0, timestamp1;
    std::vector<double> ground_truth0, ground_truth1;
    dataset.readItemWithGroundTruth(513, rgb_file0, depth_file0, timestamp0, ground_truth0);
    dataset.readItemWithGroundTruth(514, rgb_file1, depth_file1, timestamp1, ground_truth1);
    cv::Mat rgb0 = cv::imread(rgb_file0, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat rgb1 = cv::imread(rgb_file1, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth0 = cv::imread(depth_file0, CV_LOAD_IMAGE_UNCHANGED);

    Quaterniond qwc0(ground_truth0[6], ground_truth0[3], ground_truth0[4], ground_truth0[5]);
    qwc0.normalize();
    SE3d Twc0(qwc0.toRotationMatrix(), Vector3d(ground_truth0[0], ground_truth0[1], ground_truth0[2]));

    Quaterniond qwc1(ground_truth1[6], ground_truth1[3], ground_truth1[4], ground_truth1[5]);
    qwc1.normalize();
    SE3d Twc1(qwc1.toRotationMatrix(), Vector3d(ground_truth1[0], ground_truth1[1], ground_truth1[2]));

    SE3d T_0_from_1 = Twc0.inverse() * Twc1;

    std::cout << "Timestamp: " << std::setprecision(16) << timestamp0 << " " << timestamp1 << std::endl;

    AbstractCamera::Ptr camera = std::static_pointer_cast<AbstractCamera>(PinholeCamera::create(argv[1]));
    Config::file_name_ = std::string(argv[2]);
    int width = camera->width();
    int height = camera->height();
    int nlevel = Config::imageNLevel();
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    double fast_min_eigen = Config::fastMinEigen();

    cv::Mat K = camera->K();
    cv::Mat DistCoef = camera->D();

    Frame::Ptr frame0 = Frame::create(rgb0, 0, camera);
    Frame::Ptr frame1 = Frame::create(rgb1, 0, camera);
    frame0->setPose(Matrix3d::Identity(), Vector3d::Zero());
    frame1->setPose(Matrix3d::Identity(), Vector3d::Zero());

    std::vector<Corner> corners, old_corners;
    FastDetector::Ptr fast_detector = FastDetector::create(width, height, 8, nlevel, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    fast_detector->detect(frame0->images(), corners, old_corners, 200, fast_min_eigen);

    cv::Mat kps_img;
    std::vector<cv::KeyPoint> keypoints;
    std::for_each(corners.begin(), corners.end(), [&](Corner corner){
      cv::KeyPoint kp(corner.x, corner.y, 0);
      keypoints.push_back(kp);
    });
    cv::drawKeypoints(rgb0, keypoints, kps_img);

    fast_detector->drawGrid(kps_img, kps_img);
    cv::imshow("KeyPoints detectByImage", kps_img);
    cv::waitKey(0);

    for(Corner corner : corners)
    {
        int u = corner.x;
        int v = corner.y;
        uint16_t depth = depth0.at<uint16_t>(v, u);
        if(depth == 0)
            continue;

        Vector2d px_ref(u, v);
        Vector3d pt = frame1->cam_->lift(px_ref);
        pt *= depth/5000.0/pt[2];

        ssvo::MapPoint::Ptr mpt = ssvo::MapPoint::create(pt);
        Feature::Ptr feature_ref = Feature::create(px_ref, pt.normalized(), 0, mpt);

        frame0->addFeature(feature_ref);
    }

    LOG(INFO) << "Start Alignmnet";

    frame1->setPose(Matrix3d::Identity(), Vector3d(0.0,0.0,0.0));//0.01, 0.02, 0.03));
    AlignSE3 align(true, true);
    align.run(frame0, frame1, frame0->images().size()-1, 30, 1e-8);

    double t0 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++)
    {
        frame1->setPose(Matrix3d::Identity(), Vector3d(0.0,0.0,0.0));
        AlignSE3 align(false, false);
        align.run(frame0, frame1, 3, 0, 30, 1e-6);
    }
    std::cout << "Time(ms): " << (cv::getTickCount()-t0)/cv::getTickFrequency() << std::endl;

//    align_by_ceres(frame0, frame1, 0);

    SE3d T_w_from_c = frame1->pose();
    AngleAxisd aa0; aa0 = T_w_from_c.rotationMatrix();
    AngleAxisd aa1; aa1 = T_0_from_1.rotationMatrix();
    LOG(INFO) << "Aligen se3: " << T_w_from_c.translation().transpose() << " "
              << aa0.axis().transpose() <<  " " << aa0.angle();
    LOG(INFO) << "Ground se3: " << T_0_from_1.translation().transpose() << " "
              << aa1.axis().transpose() <<  " " << aa1.angle();
    LOG(INFO) << "\n"  << T_w_from_c.matrix();
    LOG(INFO) << "\n"  << T_0_from_1.matrix();
    double dist = T_0_from_1.translation().norm() - T_w_from_c.translation().norm();
    LOG(INFO) << "\n"  << dist << " " << dist/T_0_from_1.translation().norm();

    return 0;
}
