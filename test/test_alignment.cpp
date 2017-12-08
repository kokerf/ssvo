#include <fstream>
#include "feature_detector.hpp"
#include "alignment.hpp"
#include "utils.hpp"
#include "dataset.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 4) << "Usge: ./test_alignment configfile path_to_sequence path_to_association";

    TUMDataReader dataset(argv[2], argv[3]);

    std::string rgb_file0, rgb_file1, depth_file0, depth_file1;
    double timestamp0, timestamp1;
    dataset.readItemByIndex(10, rgb_file0, depth_file0, timestamp0);
    dataset.readItemByIndex(12, rgb_file1, depth_file1, timestamp1);
    cv::Mat rgb0 = cv::imread(rgb_file0, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat rgb1 = cv::imread(rgb_file1, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth0 = cv::imread(depth_file0, CV_LOAD_IMAGE_UNCHANGED);

    Config::FileName = std::string(argv[1]);
    int width = Config::imageWidth();
    int height = Config::imageHeight();
    int level = Config::imageTopLevel();
    int image_border = Config::imageBorder();
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    double fast_min_eigen = Config::fastMinEigen();

    cv::Mat K = Config::cameraIntrinsic();
    cv::Mat DistCoef = Config::cameraDistCoef();

    Camera::Ptr camera = Camera::create(Config::imageWidth(), Config::imageHeight(), K, DistCoef);

    Frame::Ptr frame0 = Frame::create(rgb0, 0, camera);
    Frame::Ptr frame1 = Frame::create(rgb1, 0, camera);
    frame0->setPose(Matrix3d::Identity(), Vector3d::Zero());
    frame1->setPose(Matrix3d::Identity(), Vector3d::Zero());

    std::vector<Corner> corners, old_corners;
    FastDetector::Ptr fast_detector = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    fast_detector->detect(frame0->image(), corners, old_corners, 100, fast_min_eigen);

    cv::Mat kps_img;
    std::vector<cv::KeyPoint> keypoints;
    std::for_each(corners.begin(), corners.end(), [&](Corner corner){
      cv::KeyPoint kp(corner.x, corner.y, 0);
      keypoints.push_back(kp);
    });
    cv::drawKeypoints(rgb0, keypoints, kps_img);

    fast_detector->drawGrid(kps_img, kps_img);
    cv::imshow("KeyPoints detectByImage", kps_img);
    cv::waitKey(0);

    for(Corner corner : corners)
    {
        int u = corner.x;
        int v = corner.y;
        uint16_t depth = depth0.at<uint16_t>(v, u);
        if(depth == 0)
            continue;

        Vector2d px_ref(u, v);
        Vector3d pt = frame1->cam_->lift(px_ref);
        pt *= depth/5000.0/pt[2];

        ssvo::MapPoint::Ptr mpt = ssvo::MapPoint::create(pt);
        Feature::Ptr feature_ref = Feature::create(px_ref, pt.normalized(), 0, mpt);

        frame0->addFeature(feature_ref);
    }

//    frame1->setPose(Matrix3d::Identity(), Vector3d(0.01, 0.02, 0.03));
    AlignSE3 align(30, 1e-8);
    align.run(frame0, frame1);
    LOG(INFO) << frame1->pose().log().transpose();

    return 0;
}
