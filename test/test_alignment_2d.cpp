#include <opencv2/opencv.hpp>
#include "feature_alignment.hpp"
#include "utils.hpp"

using namespace ssvo;

bool align2D(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate)
{
    const int halfpatch_size_ = AlignPatch8x8::HalfSize;
    const int patch_size_ = AlignPatch8x8::Size;
    const int patch_area_ = AlignPatch8x8::Area;
    bool converged=false;

#ifdef _MSC_VER 
    float ref_patch_dx[patch_area_];
    float ref_patch_dy[patch_area_];

#else
    // compute derivative of template and prepare inverse compositional
    float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
    float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
#endif

    Matrix3f H; H.setZero();

    // compute gradient and hessian
    const int ref_step = patch_size_+2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    for(int y=0; y<patch_size_; ++y)

    {
        uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
        {
            Vector3f J;
            J[0] = 0.5 * (it[1] - it[-1]);
            J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
            J[2] = 1;
            *it_dx = J[0];
            *it_dy = J[1];
            H += J*J.transpose();
        }
    }

    Matrix3f Hinv = H.inverse();
    float mean_diff = 0;

    // Compute pixel location in new image:
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    // termination condition
    const float min_update_squared = 0.01*0.01;
    const int cur_step = cur_img.step.p[0];
//  float chi2 = 0;
    Vector3f update; update.setZero();
    for(int iter = 0; iter<n_iter; ++iter)
    {
        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
            break;

        if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
            return false;

        // compute interpolation weights
        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;

        // loop through search_patch, interpolate
        uint8_t* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
//    float new_chi2 = 0.0;
        Vector3f Jres; Jres.setZero();
        for(int y=0; y<patch_size_; ++y)
        {
            uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
            for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                float res = search_pixel - *it_ref + mean_diff;

                Jres[0] -= res*(*it_ref_dx);
                Jres[1] -= res*(*it_ref_dy);
                Jres[2] -= res;
//        new_chi2 += res*res;
            }
        }

        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];


#if SUBPIX_VERBOSE
        cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif
        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
#if SUBPIX_VERBOSE
            cout << "converged." << endl;
#endif
            converged=true;
            break;
        }
    }

    cur_px_estimate << u, v;
    return converged;
}

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    //FLAGS_log_dir = std::string(getcwd(NULL,0))+"/../log";
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 2) << " Usage: ./test_alignment_2d image";

    cv::Mat rgb = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    LOG_ASSERT(!rgb.empty()) << "Can not open image: " << argv[1];

    cv::Mat gray;
    cv::cvtColor(rgb, gray, CV_RGB2GRAY);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 30, 0.1, 5);
    LOG_ASSERT(!corners.empty()) << "No corners detected!";

    cv::Mat show = rgb.clone();
    for(size_t i = 0; i < corners.size(); i++)
    {
        cv::RNG rng(i);
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(show, corners[i], 4, color, 2);
    }

    cv::imshow("Corners", show);
    cv::waitKey(0);

    cv::Mat gray_gx;
    cv::Mat gray_gy;
    cv::Sobel(gray, gray_gx, CV_64FC1, 1, 0, 3, 1.0/8);
    cv::Sobel(gray, gray_gy, CV_64FC1, 0, 1, 3, 1.0/8);

    cv::imshow("gx", gray_gx/255);
    cv::imshow("gy", gray_gy/255);

    cv::Mat noise = gray.clone();
    float mu = 20;
    float sigma = 3;
    cv::randn(noise, mu, sigma);
    cv::add(noise, gray, noise);
    std::cout << "noise with mu: " << mu << ", sigma: " << sigma << std::endl;
    cv::imshow("No Noise", gray);
    cv::imshow("Noise", noise);
    cv::waitKey(0);

    cv::Point2f p = corners[0];

    const int patch_size = AlignPatch8x8::Size;
    const int patch_size_with_border = AlignPatch8x8::SizeWithBorder;
    Matrix<float, patch_size_with_border, patch_size_with_border, RowMajor> patch_ref_with_border;
    Matrix<uchar, Dynamic, Dynamic, RowMajor> gray_eigen = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >(gray.data, gray.rows, gray.cols);
    utils::interpolateMat<uchar, float, patch_size_with_border>(gray_eigen, patch_ref_with_border, p.x, p.y);
    Matrix<uchar, patch_size_with_border, patch_size_with_border, RowMajor> patch_ref_with_border_u8 = patch_ref_with_border.cast<uchar>();
    Matrix<uchar, patch_size, patch_size, RowMajor> patch_ref_u8 = patch_ref_with_border_u8.block<patch_size, patch_size>(1,1);
    std::cout << "img: \n" << patch_ref_with_border << std::endl;

    Matrix<double, patch_size, patch_size, RowMajor> patch_cur;

    const int max_iter = 30;
    const float EPS = 1E-2f;
    const int N = 1000;
    const float scale = N/1000.0;
    Eigen::Vector3d px_error(-4.1, -3.3, 0);
    {
        double t0 = (double) cv::getTickCount();
        bool converged0 = false;
        Eigen::Vector3d estimate0;
        for(int i = 0; i < N; i++)
        {
            estimate0 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            converged0 = AlignPatch8x8::align2DI(gray, patch_ref_with_border, estimate0, max_iter, EPS);
        }

        double t1 = (double) cv::getTickCount();

        Eigen::Vector3d estimate1;
        bool converged1 = false;
        for(int i = 0; i < N; i++)
        {
            estimate1 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            converged1 = AlignPatch8x8::align2DI(noise, patch_ref_with_border, estimate1, max_iter, EPS);
        }
        double t2 = (double) cv::getTickCount();

        Eigen::Vector2d estimate2;
        bool converged2 = false;
        for(int i = 0; i < N; i++)
        {
            estimate2 = Eigen::Vector2d(p.x, p.y) + px_error.head<2>();
            converged2 = align2D(gray, patch_ref_with_border_u8.data(), patch_ref_u8.data(), max_iter, estimate2);
        }
        double t3 = (double) cv::getTickCount();

        Eigen::Vector2d estimate3;
        bool converged3 = false;
        for(int i = 0; i < N; i++)
        {
            estimate3 = Eigen::Vector2d(p.x, p.y) + px_error.head<2>();
            converged3 = align2D(noise, patch_ref_with_border_u8.data(), patch_ref_u8.data(), max_iter, estimate3);
        }
        double t4 = (double) cv::getTickCount();

        std::cout << "\nTruePose: [" << p.x << ", " << p.y << "]" << std::endl;
        std::cout << "================\n"
                  << "AlignPatch::align2DI gray\n"
                  << "Estiamte: [" << estimate0.transpose() << "]\n"
                  << "Converged: " << converged0 << " "
                  << ", Time(ms): " << (t1 - t0) / cv::getTickFrequency() / scale << std::endl;

        std::cout << "================\n"
                  << "AlignPatch::align2DI noise\n"
                  << "Estiamte: [" << estimate1.transpose() << "]\n"
                  << "Converged: " << converged1 << " "
                  << ", Time(ms): " << (t2 - t1) / cv::getTickFrequency() / scale << std::endl;

        std::cout << "================\n"
                  << "align gray\n"
                  << "Estiamte: [" << estimate2.transpose() << "]\n"
                  << "Converged: " << converged2 << " "
                  << ", Time(ms): " << (t3 - t2) / cv::getTickFrequency() / scale << std::endl;

        std::cout << "================\n"
                  << "align noise\n"
                  << "Estiamte: [" << estimate3.transpose() << "]\n"
                  << "Converged: " << converged3 << " "
                  << ", Time(ms): " << (t4 - t3) / cv::getTickFrequency() / scale << std::endl;
    }

    std::cout << std::endl;
    {
        Matrix<float, AlignPatch8x8::Area, 1> patch_ref, patch_ref_gx, patch_ref_gy;
        double t0 = (double) cv::getTickCount();
        bool converged0 = false;
        Eigen::Vector3d estimate0;
        for(int i = 0; i < N; i++)
        {
            estimate0 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            utils::interpolateMat<uchar, float, patch_size_with_border>(gray, patch_ref_with_border, p.x, p.y);
            converged0 = AlignPatch8x8::align2DI(gray, patch_ref_with_border, estimate0, max_iter, EPS);
        }
        double t1 = (double) cv::getTickCount();

        bool converged1 = false;
        Eigen::Vector3d estimate1;
        for(int i = 0; i < N; i++)
        {
            estimate1 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            utils::interpolateMat<uchar, float, patch_size>(gray, patch_ref, patch_ref_gx, patch_ref_gy, p.x, p.y);
            converged1 = AlignPatch8x8::align2DI(gray, patch_ref, patch_ref_gx, patch_ref_gy, estimate1, max_iter, EPS);
        }
        double t2 = (double) cv::getTickCount();

        for(int i = 0; i < N; i++)
        {
            estimate0 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            converged0 = AlignPatch8x8::align2DI(gray, patch_ref_with_border, estimate0, max_iter, EPS);
        }
        double t3 = (double) cv::getTickCount();

        for(int i = 0; i < N; i++)
        {
            estimate1 = Eigen::Vector3d(p.x, p.y, 0) + px_error;
            converged1 = AlignPatch8x8::align2DI(gray, patch_ref, patch_ref_gx, patch_ref_gy, estimate1, max_iter, EPS);
        }
        double t4 = (double) cv::getTickCount();

        std::cout << "================\n"
                  << "impute patch_with_border\n"
                  << "Estiamte: [" << estimate0.transpose() << "]\n"
                  << "Converged: " << converged0 << " "
                  << ", Time(ms): " << (t1 - t0) / cv::getTickFrequency() / scale
                  << "(with interpolate), " << (t3 - t2) / cv::getTickFrequency() / scale << std::endl;

        std::cout << "================\n"
                  << "impute I gx gy\n"
                  << "Estiamte: [" << estimate1.transpose() << "]\n"
                  << "Converged: " << converged1 << " "
                  << ", Time(ms): " << (t2 - t1) / cv::getTickFrequency() / scale
                  << "(with interpolate), " << (t4 - t3) / cv::getTickFrequency() / scale << std::endl;
    }

    getchar();

    return 0;
}