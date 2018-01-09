#include "utils.hpp"
#include "feature_alignment.hpp"

namespace ssvo{

template<>
const Pattern<float, 49, 13> AlignPattern<13, 49, 3>::pattern_(pattern5);

//
// Align Patch
//
bool Align2DI::run(const cv::Mat &image_cur,
                   const Matrix<float, PatchSize + 2, PatchSize + 2, RowMajor> &patch_ref_with_border,
                   Vector3d &estimate,
                   const int max_iterations,
                   const double epslion)
{
    logs_.clear();
    const double min_update_squared = epslion*epslion;
    bool converged = false;

    //! get jacobian
    float ref_patch_gx[PatchArea] = {0.0};
    float ref_patch_gy[PatchArea] = {0.0};
    const int stride = PatchSize + 2;
    const float* patch_ref_with_border_ptr = patch_ref_with_border.data() + stride + 1;
    Matrix3f H; H.setZero();
    for(int y = 0, i = 0; y < PatchSize; ++y)
    {
        const float* patch_ptr = patch_ref_with_border_ptr + y*stride;
        for(int x = 0; x < PatchSize; ++x, ++patch_ptr, ++i)
        {
            Vector3f J;
            J[0] = 0.5f * (patch_ptr[1] - patch_ptr[-1]);
            J[1] = 0.5f * (patch_ptr[stride] - patch_ptr[-stride]);
            J[2] = 1;
            H += J * J.transpose();
            ref_patch_gx[i] = J[0];
            ref_patch_gy[i] = J[1];
        }
    }

    if(H.determinant() < 1e-10)
    {
        return false;
    }

    Matrix3f Hinv = H.inverse();

    Vector3f update(0, 0, 0);

    const int border = HalfPatchSize + 1;
    const int u_min = border;
    const int v_min = border;
    const int u_max = image_cur.cols - border;
    const int v_max = image_cur.rows - border;
    float u = (float)estimate[0];
    float v = (float)estimate[1];
    float idiff = (float)estimate[2];
    for(int iter = 0; iter < max_iterations; iter++)
    {

        if(u < u_min || v < v_min || u >= u_max || v >= v_max)
        {
            LOG_IF(INFO, verbose_) << "WARNING! The estimate pixel location is out of the scope!";
            return false;
        }

        // compute interpolation weights
        Matrix<float, PatchSize, PatchSize, RowMajor> patch_cur;
        utils::interpolateMat<uchar, float, PatchSize>(image_cur, patch_cur, u, v);
        float* patch_cur_ptr = patch_cur.data();
        Vector3f Jres(0, 0, 0);
        for(int y = 0, i = 0; y < PatchSize; ++y)
        {
            const float* cur_ptr = patch_cur_ptr + y*PatchSize;
            const float* ref_ptr = patch_ref_with_border_ptr + y*stride;
            for(int x = 0; x < PatchSize; ++x, ++cur_ptr, ++ref_ptr, ++i)
            {
                float res = *cur_ptr - *ref_ptr + idiff;
                Jres[0] += ref_patch_gx[i] * res;
                Jres[1] += ref_patch_gy[i] * res;
                Jres[2] += res;
            }
        }

        //! update
        update = Hinv * Jres;
        u -= update[0];
        v -= update[1];
        idiff -= update[2];

        if(verbose_)
        {
            using std::to_string;
            std::string log = " Iter:" + to_string(iter) +
                " estimate: [" + to_string(u) + ", " + to_string(v) + ", " + to_string(idiff) + "]\n";
            logs_.push_back(log);
        }

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    if(verbose_)
    {
        std::string output;
        std::for_each(logs_.begin(), logs_.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << output;
    }

    estimate << u, v, idiff;
    return converged;
}

//! Align patch
bool AlignP2DI::run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
                    const Matrix<float, PatternNum, 3> &patch_idxy,
                    Matrix<double, ParaNum, 1> &estimate,
                    const int max_iterations,
                    const double epslion)
{
    logs_.clear();
    const double min_update_squared = epslion*epslion;
    bool converged = false;
    estimate_ = estimate.cast<float>();

    Matrix<float, PatternNum, 1> patch_ref = patch_idxy.leftCols<1>();
    //! get jacobian
    jacbian_cache_.leftCols<2>()= patch_idxy.rightCols<2>();
    jacbian_cache_.col(2).setConstant(1);

    Hessian_ = jacbian_cache_.transpose() * jacbian_cache_;
    if(Hessian_.determinant() < 1e-10)
        return false;

    invHessian_ = Hessian_.inverse();

    Vector3f update;

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
        Matrix<float, PatchSize, PatchSize, RowMajor> patch_cur;
        Matrix<float, PatternNum, 1> residual;
        utils::interpolateMat<uchar, float, PatchSize>(image, patch_cur, u, v);
        pattern_.getPattern(patch_cur, residual);
        residual.noalias() -= patch_ref;
        residual.array() += idiff;

        Jres_ = jacbian_cache_.transpose() * residual;

        //! update
        update = invHessian_ * Jres_;
        estimate_.noalias() -= update;

        using std::to_string;
        std::string log = "iter:" + to_string(iter) + " res: " + to_string(residual.norm()/PatternNum) +
            " estimate: [" + to_string(estimate_[0]) + ", " + to_string(estimate_[1]) + ", " + to_string(estimate_[2]) + "]\n";
        logs_.push_back(log);

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    if(verbose_)
    {
        std::string output;
        std::for_each(logs_.begin(), logs_.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << output;
    }

    estimate = estimate_.cast<double>();
    return converged;
}

}