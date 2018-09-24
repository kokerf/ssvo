#ifndef _SSVO_UTILS_HPP_
#define _SSVO_UTILS_HPP_

#include <opencv2/imgproc.hpp>
#include "global.hpp"

namespace ssvo {

namespace utils {

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src, Td* dst_ptr, Td* dx_ptr, Td* dy_ptr, const double u, const double v)
{
    const int iu = floor(u);
    const int iv = floor(v);
    const float wu1 = u - iu;
    const float wu0 = 1.0 - wu1;
    const float wv1 = v - iv;
    const float wv0 = 1.0 - wv1;
    const float w_tl = wv0*wu0;
    const float w_tr = wv0*wu1;
    const float w_bl = wv1*wu0;
    const float w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 2;
    const int expand_size1 = Size + 3;
    const int start_v = iv - half_size - 1;
    const int start_u = iu - half_size - 1;

    LOG_ASSERT(start_v >= 0 && start_u >= 0 && start_v+expand_size1 <= src.cols() && start_v+expand_size1 <= src.rows())
        << " Out of image scope! image cols=" << src.cols() << " rows=" << src.rows() << ", "
        << "LT: (" << start_u << ", " << start_v << ") - "
        << "BR: (" << start_u+expand_size1 << ", " << start_v+expand_size1 << ")";

    Matrix<Td, expand_size1, expand_size1, RowMajor>
        patch_with_border = src.block(start_v, start_u, expand_size1, expand_size1).template cast<Td>();
    //! block(i,j,p,q) i-rows j-cols
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tl = w_tl * patch_with_border.block(0, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tr = w_tr * patch_with_border.block(0, 1, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_bl = w_bl * patch_with_border.block(1, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_br = w_br * patch_with_border.block(1, 1, expand_size, expand_size);

    Matrix<Td, expand_size, expand_size, RowMajor> mat_interpolate = mat_tl + mat_tr + mat_bl + mat_br;
    Matrix<Td, Size, Size, RowMajor> expand_img_x = mat_interpolate.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> expand_img_y = mat_interpolate.block(0, 1, Size, Size);

    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dst(dst_ptr);
    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dx(dx_ptr);
    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dy(dy_ptr);
    dst = mat_interpolate.block(1, 1, Size, Size);
    dx = (mat_interpolate.block(1, 2, Size, Size) - expand_img_x) * 0.5;
    dy = (mat_interpolate.block(2, 1, Size, Size) - expand_img_y) * 0.5;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src, Td* dst_ptr, Td* dx_ptr, Td* dy_ptr, const double u, const double v)
{
    assert(src.type() == cv::DataType<Ts>::type);
    Eigen::Map<Matrix<Ts, Dynamic, Dynamic, RowMajor> > src_map(src.data, src.rows, src.cols);
    const int iu = floor(u);
    const int iv = floor(v);
    const float wu1 = u - iu;
    const float wu0 = 1.0 - wu1;
    const float wv1 = v - iv;
    const float wv0 = 1.0 - wv1;
    const float w_tl = wv0*wu0;
    const float w_tr = wv0*wu1;
    const float w_bl = wv1*wu0;
    const float w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 2;
    const int expand_size1 = Size + 3;
    const int start_v = iv - half_size - 1;
    const int start_u = iu - half_size - 1;

    LOG_ASSERT(start_v >= 0 && start_u >= 0 && start_v+expand_size1 <= src.cols && start_v+expand_size1 <= src.rows)
    << " Out of image scope! image cols=" << src.cols << " rows=" << src.rows << ", "
    << "LT: (" << start_u << ", " << start_v << ") - "
    << "BR: (" << start_u+expand_size1 << ", " << start_v+expand_size1 << ")";

    Matrix<Td, expand_size1, expand_size1, RowMajor>
        patch_with_border = src_map.block(start_v, start_u, expand_size1, expand_size1).template cast<Td>();
    //! block(i,j,p,q) i-rows j-cols
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tl = w_tl * patch_with_border.block(0, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tr = w_tr * patch_with_border.block(0, 1, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_bl = w_bl * patch_with_border.block(1, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_br = w_br * patch_with_border.block(1, 1, expand_size, expand_size);

    Matrix<Td, expand_size, expand_size, RowMajor> mat_interpolate = mat_tl + mat_tr + mat_bl + mat_br;
    Matrix<Td, Size, Size, RowMajor> expand_img_x = mat_interpolate.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> expand_img_y = mat_interpolate.block(0, 1, Size, Size);

    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dst(dst_ptr);
    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dx(dx_ptr);
    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dy(dy_ptr);
    dst = mat_interpolate.block(1, 1, Size, Size);
    dx = (mat_interpolate.block(1, 2, Size, Size) - expand_img_x) * 0.5;
    dy = (mat_interpolate.block(2, 1, Size, Size) - expand_img_y) * 0.5;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src, Td* dst_ptr, const double u, const double v)
{
    const int iu = floor(u);
    const int iv = floor(v);
    const float wu1 = u - iu;
    const float wu0 = 1.0 - wu1;
    const float wv1 = v - iv;
    const float wv0 = 1.0 - wv1;
    const float w_tl = wv0*wu0;
    const float w_tr = wv0*wu1;
    const float w_bl = wv1*wu0;
    const float w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 1;
    const int start_v = iv - half_size;
    const int start_u = iu - half_size;

    LOG_ASSERT(start_v >= 0 && start_u >= 0 && start_v+expand_size <= src.cols() && start_v+expand_size <= src.rows())
    << " Out of image scope! image cols=" << src.cols() << " rows=" << src.rows() << ", "
    << "LT: (" << start_u << ", " << start_v << ") - "
    << "BR: (" << start_u+expand_size << ", " << start_v+expand_size << ")";

    Matrix<Td, expand_size, expand_size, RowMajor> patch = src.block(start_v, start_u, expand_size, expand_size).template cast<Td>();
    Matrix<Td, Size, Size, RowMajor> mat_tl = w_tl * patch.block(0, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_tr = w_tr * patch.block(0, 1, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_bl = w_bl * patch.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_br = w_br * patch.block(1, 1, Size, Size);

    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dst(dst_ptr);
    dst = mat_tl + mat_tr + mat_bl + mat_br;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src, Td* dst_ptr, const double u, const double v)
{
    assert(src.type() == cv::DataType<Ts>::type);
    Eigen::Map<Matrix<Ts, Dynamic, Dynamic, RowMajor> > src_map(src.data, src.rows, src.cols);
    const int iu = floor(u);
    const int iv = floor(v);
    const float wu1 = u - iu;
    const float wu0 = 1.0 - wu1;
    const float wv1 = v - iv;
    const float wv0 = 1.0 - wv1;
    const float w_tl = wv0*wu0;
    const float w_tr = wv0*wu1;
    const float w_bl = wv1*wu0;
    const float w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 1;
    const int start_v = iv - half_size;
    const int start_u = iu - half_size;

    LOG_ASSERT(start_v >= 0 && start_u >= 0 && start_v+expand_size <= src.cols && start_v+expand_size <= src.rows)
    << " Out of image scope! image cols=" << src.cols << " rows=" << src.rows << ", "
    << "LT: (" << start_u << ", " << start_v << ") - "
    << "BR: (" << start_u+expand_size << ", " << start_v+expand_size << ")";

    Matrix<Td, expand_size, expand_size, RowMajor> patch = src_map.block(start_v, start_u, expand_size, expand_size).template cast<Td>();
    Matrix<Td, Size, Size, RowMajor> mat_tl = w_tl * patch.block(0, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_tr = w_tr * patch.block(0, 1, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_bl = w_bl * patch.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_br = w_br * patch.block(1, 1, Size, Size);

    Eigen::Map<Matrix<Td, Size, Size, RowMajor> > dst(dst_ptr);
    dst = mat_tl + mat_tr + mat_bl + mat_br;
}

//! Eigen::Matrix
template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           Matrix<Td, Size, Size, RowMajor> &dx,
                           Matrix<Td, Size, Size, RowMajor> &dy,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img.data(), dx.data(), dy.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           Matrix<Td, Size * Size, 1> &dx_vec,
                           Matrix<Td, Size * Size, 1> &dy_vec,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img_vec.data(), dx_vec.data(), dy_vec.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img_vec.data(), u, v);
}

//！ cv::Mat
template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           Matrix<Td, Size, Size, RowMajor> &dx,
                           Matrix<Td, Size, Size, RowMajor> &dy,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img.data(), dx.data(), dy.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           Matrix<Td, Size * Size, 1> &dx_vec,
                           Matrix<Td, Size * Size, 1> &dy_vec,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img_vec.data(), dx_vec.data(), dy_vec.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img_vec.data(), u, v);
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const cv::Mat &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           const double u, const double v)
{
    interpolateMat<Ts, Td, Size>(src, img.data(), u, v);
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
template <typename Ts, typename Td>
inline Td interpolateMat(const cv::Mat& mat, const double u, const double v)
{
    assert(mat.type() == cv::DataType<Ts>::type);
    int x = floor(u);
    int y = floor(v);
    float wx1 = u - x;
    float wx0 = 1.0 - wx1;
    float wy1 = v - y;
    float wy0 = 1.0 - wy1;

    const int stride = mat.step[0]/mat.step[1];
    const Ts* ptr = mat.ptr<Ts>(y) + x;
    return (wx0*wy0)*ptr[0] + (wx1*wy0)*ptr[1] + (wx0*wy1)*ptr[stride] + (wx1*wy1)*ptr[stride + 1];
}

//! ===========================================================================================

template<class T>
inline T getMedian(std::vector<T> &data_vec)
{
    assert(!data_vec.empty());
    typename std::vector<T>::iterator it = data_vec.begin()+floor(data_vec.size()/2);
    std::nth_element(data_vec.begin(), it, data_vec.end());
    return *it;
}

template <typename T>
inline double normal_distribution(T x, T mu, T sigma)
{
    static const double inv_sqrt_2pi = 0.3989422804014327f;
    double a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5 * a * a);
}

template <typename T>
inline void reduceVecor(std::vector<T>& vecs, const std::vector<bool>& inliers)
{
    size_t size = inliers.size();
    assert(size == vecs.size());

    typename std::vector<T>::iterator vecs_iter = vecs.begin();
    size_t idx = 0;
    for(;vecs_iter!=vecs.end();)
    {
        if(!inliers[idx])
        {
            inliers[idx] = inliers[--size];
            *vecs_iter = vecs.back();
            vecs.pop_back();
            continue;
        }
        idx++;
        vecs_iter++;
    }
}

inline double reprojectError(const Vector2d &fn, const SE3d &Tcw, const Vector3d &pw)
{
    Vector3d xyz_cur(Tcw*pw);
    Vector2d resdual = fn - xyz_cur.head<2>()/xyz_cur[2];
    return resdual.squaredNorm();
}

//! functions not using  template
void kltTrack(const ImgPyr& imgs_ref, const ImgPyr& imgs_cur, const cv::Size win_size,
              const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur,
              std::vector<bool> &status, cv::TermCriteria termcrit, bool track_forward = false, bool verbose = false);

bool triangulate(const Matrix3d &R_cr, const Vector3d &t_cr, const Vector3d &fn_r, const Vector3d &fn_c, double &d_ref);

namespace Fundamental
{

bool findFundamentalMat(const std::vector<cv::Point2d> &pts_prev, const std::vector<cv::Point2d> &pts_next,
                       Matrix3d &F, std::vector<bool> &inliers,
                       const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

void computeErrors(const cv::Point2d &p1, const cv::Point2d &p2, const Matrix3d &F21, double &err1, double &err2);

double computeErrorSquared(const Vector3d &p1, const Vector3d &p2, const SE3d &T, const Vector2d &p);

void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, Matrix3d& T);

bool run8point(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next,
               Matrix3d& F, const bool bE = false);

bool runRANSAC(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next,
                Matrix3d& F, std::vector<bool> &inliers,
                const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

void decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t);

Matrix3d computeE12(const SE3d &Tc1w, const SE3d &Tc2w);

}//! namespace Fundamental

}//! namespace utils

}//! namespace ssvo

#endif //_SSVO_UTILS_HPP_
