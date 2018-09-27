#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "utils.hpp"
#include "optimizer.hpp"

using namespace std;
using namespace ssvo;

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);
    if(argc != 3)
    {
        std::cout << " Usage: ./test_optimizer calib_file config_file" << std::endl;
        return -1;
    }

    AbstractCamera::Ptr cam = std::static_pointer_cast<AbstractCamera>(PinholeCamera::create(argv[1]));
    Config::file_name_ = std::string(argv[2]);
    int width = cam->width();
    int height = cam->height();
    cv::Mat K = cam->K();
    cv::Mat DistCoef = cam->D();

    std::cout << "K: \n" << K << std::endl;
    K.at<double>(0,0) += 0.5;
    K.at<double>(0,2) -= 0.5;
    K.at<double>(1,1) += 1.5;
    K.at<double>(1,2) += 1.0;
    std::cout << "K with noise: \n" << K << std::endl;
    cv::Mat img = cv::Mat(width, height, CV_8UC1);

    int nlevels = Config::imageNLevels();
    double scale_factor = Config::imageScaleFactor();
    int image_border = 8;
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    FastDetector::Ptr detector = FastDetector::create(width, height, image_border, nlevels, scale_factor, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    Frame::initScaleParameters(detector);

    Frame::Ptr frame1 = Frame::create(img, 0, cam);
    Frame::Ptr frame2 = Frame::create(img, 0, cam);
    Frame::Ptr frame3 = Frame::create(img, 0, cam);

    frame1->setPose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0,0.0,0.0));
    frame2->setPose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.1,0.0,0.0));
    frame3->setPose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.2,0.0,0.0));

    Eigen::Vector3d pose(10, 3, 30);
    MapPoint::Ptr mpt = MapPoint::create(pose);

    Eigen::Vector2d px1 = cam->project(frame1->Tcw() * pose);
    Eigen::Vector2d px2 = cam->project(frame2->Tcw() * pose);
    Eigen::Vector2d px3 = cam->project(frame3->Tcw() * pose);
    std::cout << "px1: " << px1.transpose() << std::endl;
    std::cout << "px2: " << px2.transpose() << std::endl;
    std::cout << "px3: " << px3.transpose() << std::endl;

    Feature::Ptr ft1 = Feature::create(px1, frame1->cam_->lift(px1), 0);
    Feature::Ptr ft2 = Feature::create(px2, frame2->cam_->lift(px2), 0);
    Feature::Ptr ft3 = Feature::create(px3, frame3->cam_->lift(px3), 0);
    frame1->addMapPointFeatureMatch(mpt, ft1);
    frame2->addMapPointFeatureMatch(mpt, ft2);
    frame3->addMapPointFeatureMatch(mpt, ft3);

    KeyFrame::Ptr kf1 = KeyFrame::create(frame1);
    KeyFrame::Ptr kf2 = KeyFrame::create(frame2);
    KeyFrame::Ptr kf3 = KeyFrame::create(frame3);

    mpt->addObservation(kf1, 0);
    mpt->addObservation(kf2, 0);
    mpt->addObservation(kf3, 0);
    Eigen::Vector3d pose_noise(pose[0]+0.011, pose[1]-0.001, pose[2]+2);
    mpt->setPose(pose_noise);

    double rpj_err_pre = 0;
    rpj_err_pre += utils::reprojectError(ft1->fn_.head<2>(), kf1->Tcw(), mpt->pose());
    rpj_err_pre += utils::reprojectError(ft2->fn_.head<2>(), kf2->Tcw(), mpt->pose());
    rpj_err_pre += utils::reprojectError(ft3->fn_.head<2>(), kf3->Tcw(), mpt->pose());

    double t0 = (double)cv::getTickCount();
    Optimizer::refineMapPoint(mpt, 10, false, false);
    double t1 = (double)cv::getTickCount();

    double rpj_err_aft = 0;
    rpj_err_aft += utils::reprojectError(ft1->fn_.head<2>(), kf1->Tcw(), mpt->pose());
    rpj_err_aft += utils::reprojectError(ft2->fn_.head<2>(), kf2->Tcw(), mpt->pose());
    rpj_err_aft += utils::reprojectError(ft3->fn_.head<2>(), kf3->Tcw(), mpt->pose());

    Eigen::Vector2d px11 = kf1->cam_->project(kf1->Tcw() * pose);
    Eigen::Vector2d px21 = kf1->cam_->project(kf2->Tcw() * pose);
    Eigen::Vector2d px31 = kf1->cam_->project(kf3->Tcw() * pose);
    std::cout << "px1: " << px11.transpose() << std::endl;
    std::cout << "px2: " << px21.transpose() << std::endl;
    std::cout << "px3: " << px31.transpose() << std::endl;

    std::cout << "nose pose: " << pose_noise.transpose()
              << "\ntrue pose: " << pose.transpose()
              << "\nestm pose: " << mpt->pose().transpose() << std::endl;
    std::cout << "Reproject Error changed from " << rpj_err_pre << " to " << rpj_err_aft << " time: "
              << (t1-t0)*1000/cv::getTickFrequency() << "ms" << std::endl;

    return 0;
}

