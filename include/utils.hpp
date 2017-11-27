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

template<typename Ts, typename Td, int size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, size, size, RowMajor> &img,
                           Matrix<Td, size, size, RowMajor> &dx,
                           Matrix<Td, size, size, RowMajor> &dy,
                           const double u,
                           const double v) {
    const int iu = floorf(u);
    const int iv = floorf(v);
    const Td subpix_u = u - iu;
    const Td subpix_v = v - iv;
    const Td w_tl = (1.0f - subpix_u) * (1.0f - subpix_v);
    const Td w_tr = (1.0f - subpix_u) * subpix_v;
    const Td w_bl = subpix_u * (1.0f - subpix_v);
    const Td w_br = 1.0f - w_tl - w_tr - w_bl;

    const int half_size = size / 2;
    const int expand_size = size + 2;
    const int expand_size1 = size + 3;
    const int start_v = iv - half_size - 1;
    const int start_u = iu - half_size - 1;
    Matrix<Td, expand_size1, expand_size1, RowMajor>
        patch = src.block(start_v, start_u, expand_size1, expand_size1).template cast<Td>();
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tl = w_tl * patch.block(0, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_tr = w_tr * patch.block(0, 1, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_bl = w_bl * patch.block(1, 0, expand_size, expand_size);
    Matrix<Td, expand_size, expand_size, RowMajor> mat_br = w_br * patch.block(1, 1, expand_size, expand_size);

    Matrix<Td, expand_size, expand_size, RowMajor> mat_interpolate = mat_tl + mat_tr + mat_bl + mat_br;
    Matrix<Td, size, size, RowMajor> expand_img_x = mat_interpolate.block(1, 0, size, size);
    Matrix<Td, size, size, RowMajor> expand_img_y = mat_interpolate.block(0, 1, size, size);
    img = mat_interpolate.block(1, 1, size, size);
    dx = (mat_interpolate.block(1, 2, size, size) - expand_img_x) * 0.5;
    dy = (mat_interpolate.block(2, 1, size, size) - expand_img_y) * 0.5;
}

template<typename Ts, typename Td, int size>
inline void interpolateMat(const Matrix<Ts, Dynamic, Dynamic, RowMajor> &src,
                           Matrix<Td, size * size, 1> &img_vec,
                           Matrix<Td, size * size, 1> &dx_vec,
                           Matrix<Td, size * size, 1> &dy_vec,
                           const double u,
                           const double v) {
    Matrix<Td, size, size, RowMajor> img, dx, dy;
    interpolateMat<Ts, Td, size>(src, img, dx, dy, u, v);

    img_vec = Eigen::Map<Matrix<Td, size * size, 1> >(img.data());
    dx_vec = Eigen::Map<Matrix<Td, size * size, 1> >(dx.data());
    dy_vec = Eigen::Map<Matrix<Td, size * size, 1> >(dy.data());
}

}

}

#endif //_SSVO_UTILS_HPP_
