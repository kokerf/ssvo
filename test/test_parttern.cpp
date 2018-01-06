#include <opencv2/opencv.hpp>
#include "alignment.hpp"
#include "utils.hpp"

using namespace ssvo;
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = false;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_log_dir = std::string(getcwd(NULL,0))+"/../log";
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 2) << "Usge: ./test_pattern image";

    cv::Mat rgb = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat cv_mat;
    cv::cvtColor(rgb, cv_mat, CV_RGB2GRAY);

//    cv::RNG rnger(cv::getTickCount());
//    rnger.fill(cv_mat, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));

    Matrix<uchar, Dynamic, Dynamic, RowMajor> eigen_mat = Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> >((uchar*)cv_mat.data, cv_mat.rows, cv_mat.cols);
    const int num = AlignP2DI::pattern_.Num;
    const int size = AlignP2DI::pattern_.Size;
    const int size1 = AlignP2DI::pattern_.Size1;
    Matrix<double, size1, size1, RowMajor> img;
    Matrix<double, size1, size1, RowMajor> img2;
    Matrix<double, num, 3, RowMajor> patch;

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(cv_mat, corners, 30, 0.1, 5);
    LOG_ASSERT(!corners.empty()) << "No corners detected!";
    srand(cv::getTickCount());
    int id = rand()%corners.size();
    const cv::Point2f px = corners[id];

    const double x = px.x;
    const double y = px.y;
    const int N = 1000;

    Matrix<double, size, size, RowMajor> img1;
    Matrix<double, size, size, RowMajor> dx1;
    Matrix<double, size, size, RowMajor> dy1;
    utils::interpolateMat(eigen_mat, img1, dx1, dy1, x, y);
    std::cout << "eigen Mat:\n" << img1 << std::endl;
    std::cout << "eigen Mat dx:\n" << dx1 << std::endl;
    std::cout << "eigen Mat dy:\n" << dy1 << std::endl;
//    cv::imshow("img", cv_mat);
//    cv::waitKey(0);

    Vector3d estimate = Vector3d(x,y,0) + Vector3d(4,2,0);
    Vector3d estimate0 = estimate;
    Align2DI align0(false);

    Vector3d estimate1 = estimate;
    AlignP2DI align1(false);

    double t0 = cv::getTickCount();
    const int patch_size = Align2DI::PatchSize;
    Matrix<double, patch_size*patch_size, 1> img0, dx, dy;
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, double, patch_size>(eigen_mat, img0, dx, dy, x, y);
        estimate0 = estimate;
        align0.run(eigen_mat, img0, dx, dy, estimate0);
    }
    double t1 = cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, double, size1>(eigen_mat, img, x, y);
        AlignP2DI::pattern_.getPattern(img, patch);
        estimate1 = estimate;
        align1.run(eigen_mat, patch, estimate1);
    }
    double t2 = cv::getTickCount();

    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, double, patch_size>(eigen_mat, img0, dx, dy, x, y);
    }
    double t3 = cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, double, size1>(eigen_mat, img, x, y);
        AlignP2DI::pattern_.getPattern(img, patch);
    }

    double t4 = cv::getTickCount();

    double scale = N/1000;
    cout << "T(ms):\n " << (t1-t0)/cv::getTickFrequency()/scale << endl
         << " " << (t2-t1)/cv::getTickFrequency()/scale << endl
         << " " << (t3-t2)/cv::getTickFrequency()/scale << endl
         << " " << (t4-t3)/cv::getTickFrequency()/scale << endl;

    cout << x << " " << y << endl;

    std::string output;
    std::for_each(align0.logs_.begin(), align0.logs_.end(), [&](const std::string &s){output += s;});
    cout << estimate0.transpose() << endl;
    cout << output << endl;

    output.clear();
    std::for_each(align1.logs_.begin(), align1.logs_.end(), [&](const std::string &s){output += s;});
    cout << estimate1.transpose() << endl;
    cout << output << endl;

    return 0;
}

