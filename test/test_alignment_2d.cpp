#include <opencv2/opencv.hpp>
#include "alignment.hpp"
#include "utils.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = false;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_log_dir = std::string(getcwd(NULL,0))+"/../log";
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 2) << "Usge: ./test_alignment_2d image";

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
    cv::randn(noise, 20, 3);
    cv::add(noise, gray, noise);
    cv::imshow("No Noise", gray);
    cv::imshow("Noise", noise);
    cv::waitKey(0);

    cv::Point2f p = corners[0];
    Align2DI aligner(false);

    Matrix<uchar , Dynamic, Dynamic, RowMajor> eigen_gray = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)gray.data, gray.rows, gray.cols);
    Matrix<double , Dynamic, Dynamic, RowMajor> eigen_gray_gx = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> >((double*)gray_gx.data, gray_gx.rows, gray_gx.cols);
    Matrix<double , Dynamic, Dynamic, RowMajor> eigen_gray_gy = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> >((double*)gray_gy.data, gray_gy.rows, gray_gy.cols);
    Matrix<uchar , Dynamic, Dynamic, RowMajor> eigen_noise = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)noise.data, noise.rows, noise.cols);
    const int patch_size = Align2DI::PatchSize;
    Matrix<double, patch_size*patch_size, 1> img, dx, dy;
    utils::interpolateMat<uchar, double, patch_size>(eigen_gray, img, dx, dy, p.x, p.y);
    std::cout << "block: \n" << eigen_gray.block<patch_size, patch_size>((int)p.y-2, (int)p.x-2).cast<int>() << std::endl;
    std::cout << "img: \n" << Eigen::Map<Matrix<double, patch_size,patch_size, RowMajor> >(img.data()) << std::endl;
    std::cout << "dx: \n" << Eigen::Map<Matrix<double, patch_size,patch_size, RowMajor> >(dx.data()) << std::endl;
    std::cout << "dy: \n" << Eigen::Map<Matrix<double, patch_size,patch_size, RowMajor> >(dy.data()) << std::endl;

    Matrix<double, patch_size*patch_size, 1> gx, gy;
    utils::interpolateMat<double, double, patch_size>(eigen_gray_gx, gx, p.x, p.y);
    utils::interpolateMat<double, double, patch_size>(eigen_gray_gy, gy, p.x, p.y);
    std::cout << "gx: \n" << Eigen::Map<Matrix<double, patch_size,patch_size, RowMajor> >(gx.data()) << std::endl;
    std::cout << "gy: \n" << Eigen::Map<Matrix<double, patch_size,patch_size, RowMajor> >(gy.data()) << std::endl;

    double t0 = (double)cv::getTickCount();
    Eigen::Vector3d estimate;
    bool converged = false;
    for(int i = 0; i < 1000; i++)
    {
        estimate = Eigen::Vector3d(p.x-2.4, p.y+2.1,0);
        converged = aligner.run(eigen_noise, img, dx, dy, estimate);
    }

    std::cout << "================\n"
              << "TruePose: [" << p.x << ", " << p.y << "]\n"
              << "Estiamte: [" << estimate.transpose() << "]\n"
              << "Converged: " << converged << " "
              << "Time(ms): " << (cv::getTickCount()-t0)/cv::getTickFrequency() << std::endl;

    t0 = (double)cv::getTickCount();
    for(int i = 0; i < 1000; i++)
    {
        estimate = Eigen::Vector3d(p.x-6.4, p.y+5.1,0);
        converged = aligner.run(eigen_noise, img, dx, dy, estimate);
    }

    std::cout << "================\n"
              << "TruePose: [" << p.x << ", " << p.y << "]\n"
              << "Estiamte: [" << estimate.transpose() << "]\n"
              << "Converged: " << converged << " "
              << "Time(ms): " << (cv::getTickCount()-t0)/cv::getTickFrequency() << std::endl;

    return 0;
}