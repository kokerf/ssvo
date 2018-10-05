#include <opencv2/opencv.hpp>
#include "feature_alignment.hpp"
#include "utils.hpp"

using namespace ssvo;
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = true;
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

    const int num = AlignPattern::pattern_.Num;
    const int pattern_size_with_border = AlignPattern::SizeWithBorder;
    Matrix<float, pattern_size_with_border, pattern_size_with_border, RowMajor> pattern;
    Matrix<float, num, 1> pattern_patch;
    Matrix<float, num, 1> pattern_gx;
    Matrix<float, num, 1> pattern_gy;

    const int patch_size_with_border = AlignPatch8x8::SizeWithBorder;
    Matrix<float, patch_size_with_border, patch_size_with_border, RowMajor> patch;

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(cv_mat, corners, 30, 0.1, 5);
    LOG_ASSERT(!corners.empty()) << "No corners detected!";
    cv::Point2f px;
    do
    {
        srand(cv::getTickCount());
        int id = rand() % corners.size();
        px = corners[id];
    }while(px.x <= pattern_size_with_border/2 || px.y <= pattern_size_with_border/2 ||
        px.x > cv_mat.cols - pattern_size_with_border/2 || px.y > cv_mat.rows - pattern_size_with_border/2);

    const double x = px.x;
    const double y = px.y;
    const int N = 1000;

//    cv::imshow("img", cv_mat);
//    cv::waitKey(0);

    Vector3d estimate = Vector3d(x,y,0) + Vector3d(4,2,0);
    Vector3d estimate0 = estimate;
    Vector3d estimate1 = estimate;

    utils::interpolateMat<uchar, float, patch_size_with_border>(cv_mat, patch, x, y);
    utils::interpolateMat<uchar, float, pattern_size_with_border>(cv_mat, pattern, x, y);
    AlignPattern::pattern_.getPattern(pattern, pattern_patch, pattern_gx, pattern_gy);

    const int iter = 1;

    double t0 = cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, float, patch_size_with_border>(cv_mat, patch, x, y);
        estimate0 = estimate;
        AlignPatch8x8::align2DI(cv_mat, patch, estimate0, iter);
    }
    double t1 = cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, float, pattern_size_with_border>(cv_mat, pattern, x, y);
        AlignPattern::pattern_.getPattern(pattern, pattern_patch, pattern_gx, pattern_gy);
        estimate1 = estimate;
        AlignPattern::align2DI(cv_mat, pattern_patch, pattern_gx, pattern_gy, estimate1, iter);
    }
    double t2 = cv::getTickCount();

    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, float, patch_size_with_border>(cv_mat, patch, x, y);
    }
    double t3 = cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        utils::interpolateMat<uchar, float, pattern_size_with_border>(cv_mat, pattern, x, y);
        AlignPattern::pattern_.getPattern(pattern, pattern_patch);
//        AlignPattern::pattern_.getPattern(pattern, pattern_patch, pattern_gx, pattern_gy);
    }

    double t4 = cv::getTickCount();

    double scale = N/1000;
    cout << "T(ms):\n " << (t1-t0)/cv::getTickFrequency()/scale << endl
         << " " << (t2-t1)/cv::getTickFrequency()/scale << endl
         << " " << (t3-t2)/cv::getTickFrequency()/scale << endl
         << " " << (t4-t3)/cv::getTickFrequency()/scale << endl;

    cout << x << " " << y << endl;

    estimate0 = estimate;
    AlignPatch8x8::align2DI(cv_mat, patch, estimate0, 30, 0.01, true);
    cout << " Est: " <<  estimate0.transpose() << endl;

    estimate1 = estimate;
    AlignPattern::align2DI(cv_mat, pattern_patch, pattern_gx, pattern_gy, estimate1, 30, 0.01, true);
    cout << " Est: " << estimate1.transpose() << endl;

    return 0;
}

