#include "utils.hpp"
#include "image_alignment.hpp"
#include "optimizer.hpp"

namespace ssvo
{

//! min Sum { aI+b -J }
void calculateLightAffine(const cv::Mat &I, const cv::Mat &J, float &a, float &b)
{
    const int N = I.rows;
    const int patch_area = I.cols;
    assert(N == J.rows);
    assert(patch_area == J.cols);
    assert(I.type() == J.type() && I.type() == CV_32FC1);

    float sII = 0, sIJ = 0, sI = 0, sJ = 0;
    int sw = 0;
    for(int n = 0; n < N; ++n)
    {
        const float* I_ptr = I.ptr<float>(n);
        const float* J_ptr = J.ptr<float>(n);

        float res = 0;
        float pII = 0, pIJ = 0, pI = 0, pJ = 0;
        for(int i = 0; i < patch_area; ++i)
        {
            float diff = I_ptr[i] - J_ptr[i];
            res += diff*diff;

            pI += I_ptr[i];
            pJ += J_ptr[i];
            pII += I_ptr[i] * I_ptr[i];
            pIJ += I_ptr[i] * J_ptr[i];
        }

        sI += pI;
        sJ += pJ;
        sII += pII;
        sIJ += pIJ;
        sw += patch_area;
    }

    a = (sw*sIJ - sI*sJ) / (sw*sII-sI*sI);
    b = (sJ - a*sI)/sw;
}


//
// Align SE3
//

AlignSE3::AlignSE3(bool verbose, bool visible) :
    verbose_(verbose), visible_(visible)
{}

int AlignSE3::run(Frame::Ptr reference_frame,
                  Frame::Ptr current_frame,
                  int top_level,
                  int bottom_level,
                  double scale_factor,
                  int max_iterations,
                  double epslion)
{
    logs_.clear();
    double epslion_squared = epslion * epslion;
    ref_frame_ = reference_frame;
    cur_frame_ = current_frame;

    const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = ref_frame_->getMapPointFeaturesMatched();
    std::vector<Feature::Ptr> fts; fts.reserve(mpt_fts.size());
    std::vector<MapPoint::Ptr> mpts; mpts.reserve(mpt_fts.size());
    for(const auto &it : mpt_fts)
    {
        mpts.push_back(it.first);
        fts.push_back(it.second);
    }

    const size_t N = fts.size();
    LOG_ASSERT(N != 0) << " AlignSE3: Frame(" << reference_frame->id_ << ") " << " no features to track!";
    const int max_level = (int)cur_frame_->images().size() - 1;
    LOG_ASSERT(max_level >= top_level && bottom_level >= 0 && bottom_level <= top_level) << " Error align level from top " << top_level << " to bottom " << bottom_level;

    ref_feature_cache_.resize(NoChange, N);
    ref_patch_cache_.resize(N, NoChange);
    jacbian_cache_.resize(N * PatchArea, NoChange);

    T_cur_from_ref_ = cur_frame_->Tcw() * ref_frame_->pose();
    LOG_IF(INFO, verbose_) << "T_cur_from_ref_ " << T_cur_from_ref_.log().transpose();

    for(int l = top_level; l >= bottom_level; )
    {
        const int n = computeReferencePatches(l, fts, mpts);

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

            using std::to_string;
            std::string log = "Level: " + to_string(l) + " iter:" + to_string(i) + " res: " + to_string(res) + " step: "
                + to_string(se3.dot(se3));
            logs_.push_back(log);

            //! termination
            if(se3.dot(se3) < epslion_squared)
                break;
        }

        int l_next = l-1;
        while(l_next > bottom_level && Frame::scale_factors_.at(l) <= Frame::scale_factors_.at(l_next) * scale_factor) l_next--;
        l = l_next;
    }

    cur_frame_->setTcw(T_cur_from_ref_ * ref_frame_->Tcw());
    if(verbose_)
    {
        std::string output;
        std::for_each(logs_.begin(), logs_.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << output;
        LOG(INFO) << "T_cur_from_ref:\n " << T_cur_from_ref_.matrix3x4();
    }

    return count_;
}

int AlignSE3::computeReferencePatches(int level, const std::vector<Feature::Ptr> &fts, const std::vector<MapPoint::Ptr> &mpts)
{
    LOG_ASSERT(fts.size() == mpts.size()) << " Wrong input size!";
    const size_t N = fts.size();

    Vector3d ref_pose = ref_frame_->pose().translation();
    const cv::Mat ref_img = ref_frame_->getImage(level);
    const int cols = ref_img.cols;
    const int rows = ref_img.rows;
    const int border = HalfPatchSize + 1;

    const double inv_scale = Frame::inv_scale_factors_.at(level);
    const double fx = ref_frame_->cam_->fx() * inv_scale;
    const double fy = ref_frame_->cam_->fy() * inv_scale;

    int feature_counter = 0;
    for(size_t n = 0; n < N; ++n)
    {
        const MapPoint::Ptr &mpt = mpts[n];
        if(mpt == nullptr)
            continue;

        Vector2d ref_px = fts[n]->px_ * inv_scale;
        if(ref_px[0] < border || ref_px[1] < border || ref_px[0] + border > cols - 1 || ref_px[1] + border > rows - 1)
            continue;

        double depth = (mpt->pose() - ref_pose).norm();
        Vector3d ref_xyz = fts[n]->fn_;
        ref_xyz *= depth;

        ref_feature_cache_.col(feature_counter) = ref_xyz;

        //! compute jacbian(with -)
        Matrix<double, 2, 6, RowMajor> J;
        {
            const double x = ref_xyz[0];
            const double y = ref_xyz[1];
            const double z_inv = 1./ref_xyz[2];
            const double z_inv_2 = z_inv*z_inv;

            J(0,0) = -z_inv;              // -1/z
            J(0,1) = 0.0;                 // 0
            J(0,2) = x*z_inv_2;           // x/z^2
            J(0,3) = y*J(0,2);            // x*y/z^2
            J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
            J(0,5) = y*z_inv;             // y/z

            J(1,0) = 0.0;                 // 0
            J(1,1) = -z_inv;              // -1/z
            J(1,2) = y*z_inv_2;           // y/z^2
            J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
            J(1,4) = -J(0,3);             // -x*y/z^2
            J(1,5) = -x*z_inv;            // x/z
        }

        Matrix<double, PatchArea, 1> img, dx, dy;
        utils::interpolateMat<uchar, double, PatchSize>(ref_img, img, dx, dy, ref_px[0], ref_px[1]);
        img.array() *= Frame::light_affine_a_;
        img.array() += Frame::light_affine_b_;
        ref_patch_cache_.row(feature_counter) = img;
        jacbian_cache_.block(feature_counter * PatchArea, 0, PatchArea, 6) = fx * dx * J.row(0) + fy * dy * J.row(1);

        //! visiable feature counter
        feature_counter++;
    }

    return feature_counter;
}

double AlignSE3::computeResidual(int level, int N)
{
    const cv::Mat cur_img = cur_frame_->getImage(level);
    const double inv_scale = Frame::inv_scale_factors_.at(level);
    const int cols = cur_img.cols;
    const int rows = cur_img.rows;
    const int border = HalfPatchSize + 1;
    Hessian_.setZero();
    Jres_.setZero();
    double res = 0;
    count_ = 0;
    cv::Mat showimg = cv::Mat::zeros(rows, cols, CV_8UC1);
    for(int n = 0; n < N; ++n)
    {
        const Vector3d cur_xyz = T_cur_from_ref_ * ref_feature_cache_.col(n);
        const Vector2d cur_px = cur_frame_->cam_->project(cur_xyz) * inv_scale;
        if(cur_px[0] < border || cur_px[1] < border || cur_px[0] + border > cols - 1 || cur_px[1] + border > rows - 1)
            continue;

        Matrix<double, PatchArea, 1> residual;
        utils::interpolateMat<uchar, double, PatchSize>(cur_img, residual, cur_px[0], cur_px[1]);
        residual.noalias() -= ref_patch_cache_.row(n);
        Matrix<double, PatchArea, 6, RowMajor> J = jacbian_cache_.block(n*PatchArea, 0, PatchArea, 6);

        Jres_.noalias() -= J.transpose() * residual;
        Hessian_.noalias() += J.transpose() * J;

        res += residual.dot(residual) / PatchArea;
        count_++;

        if(visible_)
        {
            cv::Mat mat_double(PatchSize, PatchSize, CV_64FC1, residual.data());
            mat_double = cv::abs(mat_double);
            Vector2i start = cur_px.cast<int>() - Vector2i(HalfPatchSize, HalfPatchSize);
            Vector2i end = start + Vector2i(PatchSize, PatchSize);
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

    return res / count_;
}

namespace utils{

int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level, const float scale_factor)
{
    // Compute patch level in other image
    int search_level = 0;
    double D = A_cur_ref.determinant();
    double scale = 1.0 / (scale_factor*scale_factor);
    while(D > 3.0 && search_level < max_level)
    {
        search_level += 1;
        D *= scale;
    }
    return search_level;
}

void getWarpMatrixAffine(const AbstractCamera::Ptr &cam_ref,
                         const AbstractCamera::Ptr &cam_cur,
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
    const double length = half_patch_size * Frame::scale_factors_.at(level_ref);
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