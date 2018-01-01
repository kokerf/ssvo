#ifndef _SSVO_UTILS_HPP_
#define _SSVO_UTILS_HPP_

#include <opencv2/imgproc.hpp>
#include "global.hpp"

namespace ssvo {

namespace utils {

inline int createPyramid(const cv::Mat &img,
                         ImgPyr &img_pyr,
                         const int nlevels = 4,
                         const cv::Size min_size = cv::Size(40, 40))
{
    assert(!img.empty());

    img_pyr.resize(nlevels);
    img.copyTo(img_pyr[0]);

    for (int i = 1; i < nlevels; ++i) {
        cv::Size size(round(img_pyr[i - 1].cols >> 1), round(img_pyr[i - 1].rows >> 1));

        if (size.height < min_size.height || size.width < min_size.width) {
            img_pyr.resize(i);
            return i;
        }

        cv::resize(img_pyr[i - 1], img_pyr[i], size, 0, 0, cv::INTER_LINEAR);
    }

    return nlevels;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           Matrix<Td, Size, Size, RowMajor> &dx,
                           Matrix<Td, Size, Size, RowMajor> &dy,
                           const double u, const double v) {
    const int iu = floorf(u);
    const int iv = floorf(v);
    const double wu1 = u - iu;
    const double wu0 = 1.0 - wu1;
    const double wv1 = v - iv;
    const double wv0 = 1.0 - wv1;
    const double w_tl = wv0*wu0;
    const double w_tr = wv0*wu1;
    const double w_bl = wv1*wu0;
    const double w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 2;
    const int expand_size1 = Size + 3;
    const int start_v = iv - half_size - 1;
    const int start_u = iu - half_size - 1;
    Matrix<Td, expand_size1, expand_size1, RowMajor>
        patch = src.block(start_v, start_u, expand_size1, expand_size1).template cast<Td>();
    //! block(i,j,p,q) i-rows j-cols
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tl = w_tl * patch.block(0, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tr = w_tr * patch.block(0, 1, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_bl = w_bl * patch.block(1, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_br = w_br * patch.block(1, 1, expand_size, expand_size);

    Matrix<Td, expand_size, expand_size, RowMajor> mat_interpolate = mat_tl + mat_tr + mat_bl + mat_br;
    Matrix<Td, Size, Size, RowMajor> expand_img_x = mat_interpolate.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> expand_img_y = mat_interpolate.block(0, 1, Size, Size);
    img = mat_interpolate.block(1, 1, Size, Size);
    dx = (mat_interpolate.block(1, 2, Size, Size) - expand_img_x) * 0.5;
    dy = (mat_interpolate.block(2, 1, Size, Size) - expand_img_y) * 0.5;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size, Size, RowMajor> &img,
                           const double u, const double v) {
    const int iu = floorf(u);
    const int iv = floorf(v);
    const double wu1 = u - iu;
    const double wu0 = 1.0 - wu1;
    const double wv1 = v - iv;
    const double wv0 = 1.0 - wv1;
    const double w_tl = wv0*wu0;
    const double w_tr = wv0*wu1;
    const double w_bl = wv1*wu0;
    const double w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = Size / 2;
    const int expand_size = Size + 1;
    const int start_v = iv - half_size;
    const int start_u = iu - half_size;
    Matrix<Td, expand_size, expand_size, RowMajor> patch = src.block(start_v, start_u, expand_size, expand_size).template cast<Td>();
    Matrix<Td, Size, Size, RowMajor> mat_tl = w_tl * patch.block(0, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_tr = w_tr * patch.block(0, 1, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_bl = w_bl * patch.block(1, 0, Size, Size);
    Matrix<Td, Size, Size, RowMajor> mat_br = w_br * patch.block(1, 1, Size, Size);

    img = mat_tl + mat_tr + mat_bl + mat_br;
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           Matrix<Td, Size * Size, 1> &dx_vec,
                           Matrix<Td, Size * Size, 1> &dy_vec,
                           const double u, const double v) {
    Matrix<Td, Size, Size, RowMajor> img, dx, dy;
    interpolateMat<Ts, Td, Size>(src, img, dx, dy, u, v);

    img_vec = Eigen::Map<Matrix<Td, Size * Size, 1> >(img.data());
    dx_vec = Eigen::Map<Matrix<Td, Size * Size, 1> >(dx.data());
    dy_vec = Eigen::Map<Matrix<Td, Size * Size, 1> >(dy.data());
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, Size * Size, 1> &img_vec,
                           const double u, const double v) {
    Matrix<Td, Size, Size, RowMajor> img;
    interpolateMat<Ts, Td, Size>(src, img, u, v);
    img_vec = Eigen::Map<Matrix<Td, Size * Size, 1> >(img.data());
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
template <typename Ts, typename Td>
inline Td interpolateMat(const cv::Mat& mat, const double u, const double v)
{
    assert(mat.type() == cv::DataType<Ts>::type);
    int x = floor(u);
    int y = floor(v);
    double wx1 = u - x;
    double wx0 = 1.0 - wx1;
    double wy1 = v - y;
    double wy0 = 1.0 - wy1;

    const int stride = mat.step[0]/mat.step[1];
    const Ts* ptr = mat.ptr<Ts>(y) + x;
    return (wx0*wy0)*ptr[0] + (wx1*wy0)*ptr[1] + (wx0*wy1)*ptr[stride] + (wx1*wy1)*ptr[stride + 1];
}

template<class T>
T getMedian(std::vector<T> &data_vec)
{
    assert(!data_vec.empty());
    typename std::vector<T>::iterator it = data_vec.begin()+floor(data_vec.size()/2);
    std::nth_element(data_vec.begin(), it, data_vec.end());
    return *it;
}

template <typename T>
double normal_distribution(T x, T mu, T sigma)
{
    static const double inv_sqrt_2pi = 0.3989422804014327f;
    double a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5 * a * a);
}

template <typename T>
void reduceVecor(std::vector<T>& vecs, const std::vector<bool>& inliers)
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

//! functions not using  template
void kltTrack(const ImgPyr& imgs_ref, const ImgPyr& imgs_cur, const cv::Size win_size,
              const std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur,
              std::vector<bool> &status, bool track_forward = false, bool verbose = false);

namespace Fundamental
{

int findFundamentalMat(const std::vector<cv::Point2d> &pts_prev, const std::vector<cv::Point2d> &pts_next,
                       Matrix3d &F, std::vector<bool> &inliers,
                       const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

void computeErrors(const cv::Point2d &p1, const cv::Point2d &p2, Matrix3d &F, double &err1, double &err2);

double computeErrorSquared(const Vector3d &p1, const Vector3d &p2, const SE3d &T, const Vector2d &p);

void Normalize(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& pts_norm, Matrix3d& T);

void run8point(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next,
               Matrix3d& F, const bool bE = false);

int runRANSAC(const std::vector<cv::Point2d>& pts_prev, const std::vector<cv::Point2d>& pts_next,
              Matrix3d& F, std::vector<bool> &inliers,
              const double sigma2 = 1, const int max_iterations = 1000, const bool bE = false);

void decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t);

}//! namespace Fundamental

}//! namespace utils

}//! namespace ssvo

#endif //_SSVO_UTILS_HPP_
