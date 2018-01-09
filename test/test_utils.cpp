#include <opencv2/opencv.hpp>
#include "global.hpp"
#include "glog/logging.h"
#include "utils.hpp"

using namespace ssvo;

template <typename T, int size>
inline void interpolateMat2(cv::Mat &ref_img, cv::Mat& ref_patch,
           cv::Mat& dx, cv::Mat& dy, const double u, const double v)
{
    const int stride = ref_img.cols;
    const int  half_size = size/2;

    const int iu = floor(u);
    const int iv = floor(v);
    const double subpix_u = u - iu;
    const double subpix_v = v - iv;
    const double w_tl = (1.0f - subpix_v)*(1.0f - subpix_u);
    const double w_tr = (1.0f - subpix_v)*subpix_u;
    const double w_bl = subpix_v*(1.0f - subpix_u);
    const double w_br = 1.0f - w_tl - w_tr - w_bl;

    ref_patch = cv::Mat(size, size, CV_64FC1);
    dx = cv::Mat(size, size, CV_64FC1);
    dy = cv::Mat(size, size, CV_64FC1);
    double* ref_patch_ptr = ref_patch.ptr<double>(0);
    double* ref_dx_ptr = dx.ptr<double>(0);
    double* ref_dy_ptr = dy.ptr<double>(0);
    int pixel_counter = 0;
    for(int y = 0; y < size; ++y)
    {
        T* ref_img_ptr = ref_img.ptr<T>(iv - half_size + y) + (iu - half_size);
        for(int x = 0; x < size; ++x, ++pixel_counter, ++ref_patch_ptr, ++ref_dx_ptr, ++ref_dy_ptr, ++ref_img_ptr)
        {
            //std::cout << "ref: " << (int)ref_img_ptr[0] << " "<< (int)ref_img_ptr[1] << " " << (int)ref_img_ptr[stride] << " " << (int)ref_img_ptr[stride+1] << std::endl;
            *ref_patch_ptr = w_tl * ref_img_ptr[0] + w_tr * ref_img_ptr[1] + w_bl * ref_img_ptr[stride] + w_br * ref_img_ptr[stride+1];

            *ref_dx_ptr = ((w_tl * ref_img_ptr[1] + w_tr * ref_img_ptr[2] + w_bl * ref_img_ptr[stride+1] + w_br * ref_img_ptr[stride+2]) -
                (w_tl * ref_img_ptr[-1] + w_tr * ref_img_ptr[0] + w_bl * ref_img_ptr[stride-1] + w_br * ref_img_ptr[stride])) * 0.5;

            *ref_dy_ptr = ((w_tl * ref_img_ptr[stride] + w_tr * ref_img_ptr[1+stride] + w_bl * ref_img_ptr[stride*2] + w_br * ref_img_ptr[1+(stride*2)]) -
                (w_tl * ref_img_ptr[-stride] + w_tr * ref_img_ptr[1-stride] + w_bl * ref_img_ptr[0] + w_br * ref_img_ptr[1])) * 0.5;

        }
    }
}

template <typename T, int size>
void interpolateMat_pixel(cv::Mat &ref_img, cv::Mat& ref_patch, const double u, const double v)
{
    const int  half_size = size/2;
    ref_patch = cv::Mat(size, size, CV_64FC1);

    double* ref_patch_ptr = ref_patch.ptr<double>(0);
    for(int y = 0; y < size; ++y)
    {
        for(int x = 0; x < size; ++x, ++ref_patch_ptr)
        {
            *ref_patch_ptr = utils::interpolateMat<T, double>(ref_img, u+x-half_size, v+y-half_size);
        }
    }
}

template<typename Ts, typename Td, int Size>
inline void interpolateMat_array(const cv::Mat &src,
                                Matrix<Td, Size, Size, RowMajor> &dst,
                                Matrix<Td, Size, Size, RowMajor> &dx,
                                Matrix<Td, Size, Size, RowMajor> &dy,
                                const double u, const double v)
{
    assert(src.type() == cv::DataType<Ts>::type);

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
    const int half_expand_size = half_size+1;
    const int start_v = iv - half_expand_size;
    const int start_u = iu - half_expand_size;

    Matrix<Td, expand_size, expand_size, RowMajor> dst_with_border;

    const int src_step = src.cols;
    const Ts* src_ptr = src.ptr<Ts>(start_v) + start_u;
    Td* dst_border_ptr = dst_with_border.data();

    for(size_t r = 0; r < expand_size; ++r)
    {
        const Ts* src_row_ptr = src_ptr + src_step * r;
        for(size_t c = 0; c < expand_size; ++c, ++dst_border_ptr)
        {
            *dst_border_ptr = w_tl * src_row_ptr[c] + w_tr * src_row_ptr[c+1] + w_bl * src_row_ptr[c+src_step] + w_br * src_row_ptr[c+src_step+1];
        }
    }

    dst_border_ptr = dst_with_border.data() + expand_size + 1;
    Td* dst_ptr = dst.data();
    Td* dx_ptr = dx.data();
    Td* dy_ptr = dy.data();
    for(size_t r = 0; r < Size; ++r)
    {
        const Td* dst_border_row_ptr = dst_border_ptr + expand_size * r;
        for(size_t c = 0; c < Size; ++c, ++dst_ptr, ++dst_border_row_ptr, ++dx_ptr, ++dy_ptr)
        {
            *dst_ptr = *dst_border_row_ptr;
            *dx_ptr = 0.5*(dst_border_row_ptr[1] - dst_border_row_ptr[-1]);
            *dy_ptr = 0.5*(dst_border_row_ptr[expand_size] - dst_border_row_ptr[-expand_size]);
        }
    }
}


int main(int argc, char const *argv[])
{
    cv::RNG rnger(cv::getTickCount());
    cv::Mat cv_mat = cv::Mat(100,100,CV_8UC1);
    rnger.fill(cv_mat, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));

    cv::imshow("random image", cv_mat);

    Matrix<uchar , Dynamic, Dynamic, RowMajor> eigen_mat = Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)cv_mat.data, cv_mat.rows, cv_mat.cols);

    const int size = 8;
    Matrix<double, size, size, RowMajor> img;
    Matrix<double, size, size, RowMajor> dx;
    Matrix<double, size, size, RowMajor> dy;

    Matrix<double, size*size, 1> img_v;
    Matrix<double, size*size, 1> dx_v;
    Matrix<double, size*size, 1> dy_v;

    cv::Mat cv_img, cv_dx, cv_dy;
    cv::Mat cv_img1;

    //std::cout << "Mat:\n" << cv_mat << std::endl;

    const double y =4.0001+size/2;
    const double x =3.9999+size/2;
    std::cout << "cv image:\n" << cv_mat<< std::endl;
    std::cout << "eigen image:\n" << eigen_mat<< std::endl;
    std::cout << "eigen bolck:\n" << eigen_mat.block<size,size>(int(y-size/2),int(x-size/2)) << std::endl;

    //! =============================
    utils::interpolateMat(eigen_mat, img, dx, dy, x, y);
    std::cout << "eigen Mat:\n" << img << std::endl;
    std::cout << "eigen Mat dx:\n" << dx << std::endl;
    std::cout << "eigen Mat dy:\n" << dy << std::endl;

    utils::interpolateMat<uchar, double, size>(eigen_mat, img_v, dx_v, dy_v, x, y);
    std::cout << "eigen Vec:\n" << img_v.transpose() << std::endl;
    std::cout << "eigen Vec dx:\n" << dx_v.transpose() << std::endl;
    std::cout << "eigen Vec dy:\n" << dy_v.transpose() << std::endl;

    interpolateMat2<uchar, size>(cv_mat, cv_img, cv_dx, cv_dy, x, y);
    std::cout << "cv Mat:\n" << cv_img << std::endl;
    std::cout << "cv Mat dx:\n" << cv_dx << std::endl;
    std::cout << "cv Mat dy:\n" << cv_dy << std::endl;

    interpolateMat_pixel<uchar, size>(cv_mat, cv_img1, x, y);
    std::cout << "cv Mat pixel:\n" << cv_img1 << std::endl;

    img.setZero();
    utils::interpolateMat<uchar, double, size>(eigen_mat, img, x, y);
    std::cout << "eigen Mat:\n" << img << std::endl;

    img.setZero();
    dx.setZero();
    dy.setZero();
    interpolateMat_array<uchar, double, size>(cv_mat, img, dx, dy, x, y);
    std::cout << "eigen Mat:\n" << img << std::endl;
    std::cout << "eigen Mat dx:\n" << dx << std::endl;
    std::cout << "eigen Mat dy:\n" << dy << std::endl;

    //! ==============================
    const size_t N = 1000000;
    double t0 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat(eigen_mat, img, dx, dy, x, y);
    }

    double t1 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat<uchar, double, size>(eigen_mat, img_v, dx_v, dy_v, x, y);
    }

    double t2 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat<uchar, double, size>(cv_mat, img, dx, dy, x, y);
    }

    double t3 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat<uchar, double, size>(cv_mat, img_v, dx_v, dy_v, x, y);
    }

    double t4 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        interpolateMat2<uchar, size>(cv_mat, cv_img, cv_dx, cv_dy, x, y);
    }

    double t5 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat<uchar, double, size>(eigen_mat, img, x, y);
    }

    double t6 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        utils::interpolateMat<uchar, double, size>(cv_mat, img, x, y);
    }

    double t7 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        interpolateMat_pixel<uchar, size>(cv_mat, cv_img1, x, y);
    }

    double t8 = (double)cv::getTickCount();
    for(size_t i = 0; i < N; i++) {
        interpolateMat_array<uchar, double, size>(cv_mat, img, dx, dy, x, y);
    }

    double t9 = (double)cv::getTickCount();

    size_t scale = N / 1000;
    std::cout << "time0(ms): " << (t1-t0)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time1(ms): " << (t2-t1)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time2(ms): " << (t3-t2)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time3(ms): " << (t4-t3)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time4(ms): " << (t5-t4)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time5(ms): " << (t6-t5)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time6(ms): " << (t7-t6)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time7(ms): " << (t8-t7)/cv::getTickFrequency()/scale << std::endl;
    std::cout << "time8(ms): " << (t9-t8)/cv::getTickFrequency()/scale << std::endl;
    cv::waitKey(0);
    return 0;
}

