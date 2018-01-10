#include "utils.hpp"
#include "feature_alignment.hpp"

namespace ssvo{

const Pattern<float, 32, 8> AlignPattern::pattern_(pattern4);

//
// Align Patch
//
bool AlignPatch::align2DI(const cv::Mat &image_cur,
                          const Matrix<float, SizeWithBorder, SizeWithBorder, RowMajor> &patch_ref_with_border,
                          Vector3d &estimate,
                          const int max_iterations,
                          const double epslion,
                          const bool verbose)
{
    std::list<std::string> logs;
    const double min_update_squared = epslion*epslion;
    bool converged = false;

    //! get jacobian
    float ref_patch_gx[Area] = {0.0};
    float ref_patch_gy[Area] = {0.0};
    const int stride = Size + 2;
    const float* patch_ref_with_border_ptr = patch_ref_with_border.data() + stride + 1;
    Matrix3f H; H.setZero();
    Vector3f J(0, 0, 1);
    for(int y = 0, i = 0; y < Size; ++y)
    {
        const float* patch_ptr = patch_ref_with_border_ptr + y*stride;
        for(int x = 0; x < Size; ++x, ++patch_ptr, ++i)
        {
            J[0] = 0.5f * (patch_ptr[1] - patch_ptr[-1]);
            J[1] = 0.5f * (patch_ptr[stride] - patch_ptr[-stride]);
            H += J * J.transpose();
            ref_patch_gx[i] = J[0];
            ref_patch_gy[i] = J[1];
        }
    }

    Matrix3f Hinv = H.inverse();
    if(isinf(Hinv(0,0)) || isnan(Hinv(0,0)))
        return false;

    Vector3f update(0, 0, 0);

    const int border = HalfSize + 1;
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
            LOG_IF(INFO, verbose) << "WARNING! The estimate pixel location is out of the scope!";
            return false;
        }

        // compute interpolation weights
        Matrix<float, Size, Size, RowMajor> patch_cur;
        utils::interpolateMat<uchar, float, Size>(image_cur, patch_cur, u, v);
        float* patch_cur_ptr = patch_cur.data();
        Vector3f Jres(0, 0, 0);
        for(int y = 0, i = 0; y < Size; ++y)
        {
            const float* cur_ptr = patch_cur_ptr + y*Size;
            const float* ref_ptr = patch_ref_with_border_ptr + y*stride;
            for(int x = 0; x < Size; ++x, ++cur_ptr, ++ref_ptr, ++i)
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

        if(verbose)
        {
            using std::to_string;
            std::string log = " Iter:" + to_string(iter) +
                " estimate: [" + to_string(u) + ", " + to_string(v) + ", " + to_string(idiff) + "]\n";
            logs.push_back(log);
        }

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    if(verbose)
    {
        std::string output;
        std::for_each(logs.begin(), logs.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << "\n" << output;
    }

    estimate << u, v, idiff;
    return converged;
}

bool AlignPatch::align2DI(const cv::Mat &image_cur,
                          const Matrix<float, Area, 1> &patch_ref,
                          const Matrix<float, Area, 1> &patch_ref_gx,
                          const Matrix<float, Area, 1> &patch_ref_gy,
                          Vector3d &estimate,
                          const int max_iterations,
                          const double epslion,
                          const bool verbose)
{
    std::list<std::string> logs;
    const double min_update_squared = epslion*epslion;
    bool converged = false;

    //! get jacobian
    Matrix3f H; H.setZero();
    Vector3f J(0, 0, 1);
    for(int i = 0; i < Area; ++i)
    {
        J[0] = patch_ref_gx[i];
        J[1] = patch_ref_gy[i];
        H += J * J.transpose();
    }

    Matrix3f Hinv = H.inverse();
    if(isinf(Hinv(0,0)) || isnan(Hinv(0,0)))
        return false;

    Vector3f update(0, 0, 0);

    const int border = HalfSize + 1;
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
            LOG_IF(INFO, verbose) << "WARNING! The estimate pixel location is out of the scope!";
            return false;
        }

        // compute interpolation weights
        Matrix<float, Area, 1> patch_cur;
        utils::interpolateMat<uchar, float, Size>(image_cur, patch_cur, u, v);
        Vector3f Jres(0, 0, 0);
        for(int i = 0; i < Area; ++i)
        {
            float res = patch_cur[i] - patch_ref[i] + idiff;
            Jres[0] += patch_ref_gx[i] * res;
            Jres[1] += patch_ref_gy[i] * res;
            Jres[2] += res;
        }

        //! update
        update = Hinv * Jres;
        u -= update[0];
        v -= update[1];
        idiff -= update[2];

        if(verbose)
        {
            using std::to_string;
            std::string log = " Iter:" + to_string(iter) +
                " estimate: [" + to_string(u) + ", " + to_string(v) + ", " + to_string(idiff) + "]\n";
            logs.push_back(log);
        }

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    if(verbose)
    {
        std::string output;
        std::for_each(logs.begin(), logs.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << "\n" << output;
    }

    estimate << u, v, idiff;
    return converged;
}


//! Align patch
bool AlignPattern::align2DI(const cv::Mat &image_cur,
                            const Matrix<float, Num, 1> &patch_ref,
                            const Matrix<float, Num, 1> &patch_ref_gx,
                            const Matrix<float, Num, 1> &patch_ref_gy,
                            Vector3d &estimate,
                            const int max_iterations,
                            const double epslion,
                            const bool verbose)
{
    std::list<std::string> logs;
    const double min_update_squared = epslion*epslion;
    bool converged = false;

    Matrix3f H; H.setZero();
    Vector3f J(0, 0, 1);
    for(int i = 0; i < Num; ++i)
    {
        J[0] = patch_ref_gx[i];
        J[1] = patch_ref_gy[i];
        H += J * J.transpose();
    }

    Matrix3f Hinv = H.inverse();
    if(isinf(Hinv(0,0)) || isnan(Hinv(0,0)))
        return false;

    Vector3f update(0, 0, 0);

    const int border = HalfSize + 1;
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
            LOG_IF(INFO, verbose) << "WARNING! The estimate pixel location is out of the scope!";
            return false;
        }

        // compute interpolation weights
        Matrix<float, SizeWithBorder, SizeWithBorder, RowMajor> patch_cur_raw;
        Matrix<float, Num, 1> patch_cur;
        utils::interpolateMat<uchar, float, SizeWithBorder>(image_cur, patch_cur_raw, u, v);
        pattern_.getPattern(patch_cur_raw, patch_cur);
//        residual.noalias() -= patch_ref;
//        residual.array() += idiff;

        Vector3f Jres(0,0,0);
//        Jres[0] = patch_ref_gx.dot(residual);
//        Jres[1] = patch_ref_gy.dot(residual);
//        Jres[2] = residual.sum();
        for(int i = 0; i < Num; ++i)
        {
            float res = patch_cur[i] - patch_ref[i] + idiff;
            Jres[0] += patch_ref_gx[i] * res;
            Jres[1] += patch_ref_gy[i] * res;
            Jres[2] += res;
        }

        //! update
        update = Hinv * Jres;
        u -= update[0];
        v -= update[1];
        idiff -= update[2];

        if(verbose)
        {
            using std::to_string;
            std::string log = " Iter:" + to_string(iter) +
                " estimate: [" + to_string(u) + ", " + to_string(v) + ", " + to_string(idiff) + "]\n";
            logs.push_back(log);
        }

        if(update.dot(update) < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    if(verbose)
    {
        std::string output;
        std::for_each(logs.begin(), logs.end(), [&](const std::string &s) { output += s; });
        LOG(INFO) << "\n" << output;
    }

    estimate << u, v, idiff;
    return converged;
}

}