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
    const int feature_counter = 0;
    const int  patch_area_ = size*size;
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

int main(int argc, char const *argv[])
{
    cv::RNG rnger(cv::getTickCount());
    cv::Mat cv_mat = cv::Mat(10,10,CV_32FC1);
    rnger.fill(cv_mat, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));

    cv::imshow("random image", cv_mat);

    Matrix<float , Dynamic, Dynamic, RowMajor> eigen_mat = Map<Matrix<float, Dynamic, Dynamic, RowMajor> >((float*)cv_mat.data, cv_mat.rows, cv_mat.cols);

    const int size = 4;
    Matrix<float, size, size, RowMajor> img;
    Matrix<float, size, size, RowMajor> dx;
    Matrix<float, size, size, RowMajor> dy;

    Matrix<float, size*size, 1> img_v;
    Matrix<float, size*size, 1> dx_v;
    Matrix<float, size*size, 1> dy_v;

    cv::Mat cv_img, cv_dx, cv_dy;
    cv::Mat cv_img1;

    //std::cout << "Mat:\n" << cv_mat << std::endl;

    const double y =4.0001;
    const double x =3.9999;
    std::cout << "cv image:\n" << cv_mat<< std::endl;
    std::cout << "eigen image:\n" << eigen_mat<< std::endl;
    std::cout << "eigen bolck:\n" << eigen_mat.block<size,size>(int(y-size/2),int(x-size/2)) << std::endl;

    double t0 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++) {
        utils::interpolateMat(eigen_mat, img, dx, dy, x, y);
//        utils::interpolateMat<float, float, size>(eigen_mat, img, x, y);
    }

    double t1 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++) {
        utils::interpolateMat<float, float, size>(eigen_mat, img_v, dx_v, dy_v, x, y);
    }

    double t2 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++) {
        interpolateMat2<float, size>(cv_mat, cv_img, cv_dx, cv_dy, x, y);
    }

    double t3 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++) {
        interpolateMat_pixel<float, size>(cv_mat, cv_img1, x, y);
    }

    double t4 = (double)cv::getTickCount();
    std::cout << "eigen Mat:\n" << img << std::endl;
    std::cout << "eigen Mat dx:\n" << dx << std::endl;
    std::cout << "eigen Mat dy:\n" << dy << std::endl;
    std::cout << "eigen Vec:\n" << img_v.transpose() << std::endl;
    std::cout << "eigen Vec dx:\n" << dx_v.transpose() << std::endl;
    std::cout << "eigen Vec dy:\n" << dy_v.transpose() << std::endl;
    std::cout << "cv Mat:\n" << cv_img << std::endl;
    std::cout << "cv Mat dx:\n" << cv_dx << std::endl;
    std::cout << "cv Mat dy:\n" << cv_dy << std::endl;
    std::cout << "cv Mat1:\n" << cv_img1 << std::endl;

    std::cout << "time0(ms): " << (t1-t0)/cv::getTickFrequency() << std::endl;
    std::cout << "time1(ms): " << (t2-t1)/cv::getTickFrequency() << std::endl;
    std::cout << "time2(ms): " << (t3-t2)/cv::getTickFrequency() << std::endl;
    std::cout << "time3(ms): " << (t4-t3)/cv::getTickFrequency() << std::endl;
    cv::waitKey(0);
    return 0;
}

