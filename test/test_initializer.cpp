#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "config.hpp"

std::string ssvo::Config::FileName;

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 4) {
        std::cout << "Usge: ./test_initializer image0 image1 configflie" << std::endl;
        return -1;
    }

    cv::Mat ref_img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cur_img = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    ssvo::Config::FileName = std::string(argv[3]);

    if(ref_img.empty())
    {
        std::cout << "Can not open:" << argv[1] << std::endl;
        return -1;
    }

    if(cur_img.empty())
    {
        std::cout << "Can not open:" << argv[2] << std::endl;
        return -1;
    }

    cv::Mat K = ssvo::Config::cameraIntrinsic();
    cv::Mat DistCoef = ssvo::Config::cameraDistCoef();
    ssvo::Camera camera(ssvo::Config::imageWidth(), ssvo::Config::imageHeight(), K, DistCoef);

    cv::Mat temp;
    const int n = 1000;
    double time_accumulator = 0;
    int nlevel = 0;
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(K, DistCoef, cv::Matx33d::eye(), K, ref_img.size(), CV_32FC1, map1, map2 );
    for(int i = 0; i < n; ++i)
    {
        double t = (double)cv::getTickCount();
        //cv::fisheye::undistortImage(ref_img, temp, K, DistCoef, K);
        cv::remap(ref_img, temp, map1, map2, cv::INTER_LINEAR);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
    }
    std::cout << " took " <<  time_accumulator/((double)n)*1000.0
              << " ms (average over " << n << " trials)." << std::endl;
    std::cout << " all levels: " << nlevel << std::endl;

    cv::imshow("sss", ref_img);
    cv::imshow("sss1", temp);
    cv::waitKey(0);


    std::vector<cv::KeyPoint> kps;
    std::vector<cv::KeyPoint> kps_old;
    std::vector<cv::Point2f> pts, upts;
    std::vector<cv::Point2d> fts;
    ImgPyr img_pyr;
    ssvo::FastDetector fast_detector(1000, 3);
    ssvo::createPyramid(ref_img, img_pyr);
    fast_detector.detectByImage(img_pyr, kps, kps_old);
    cv::KeyPoint::convert(kps, pts);
    cv::undistortPoints(pts, upts, K, DistCoef);
    fts.clear();fts.reserve(kps.size());
    std::for_each(upts.begin(), upts.end(), [&](cv::Point2f& pt){fts.push_back(cv::Point2d((double)pt.x, (double)pt.y));});
    std::cout << "-- Corners in First image: " << kps.size() << std::endl;

    const int n_trials = 100;
    double time_accumulator1 = 0;
    double time_accumulator2 = 0;
    ssvo::Initializer initializer(K, DistCoef);
    for(int i = 0; i < n_trials; ++i)
    {
        double t1 = (double)cv::getTickCount();
        initializer.addFirstImage(ref_img, pts, fts);
        time_accumulator1 += ((cv::getTickCount() - t1) / cv::getTickFrequency());

        double t2 = (double)cv::getTickCount();
        initializer.addSecondImage(cur_img);
        time_accumulator2 += ((cv::getTickCount() - t2) / cv::getTickFrequency());
    }
    std::cout << " took " <<  time_accumulator1/((double)n_trials)*1000.0
              << " ms for first image and " << time_accumulator2/((double)n_trials)*1000.0 << " ms for second image(average over " << n_trials << " trials)." << std::endl;

    initializer.addFirstImage(ref_img, pts, fts);
    int succeed = initializer.addSecondImage(cur_img);
    std::vector<cv::Point2f> pts_ref, pts_cur;
    initializer.getTrackedPoints(pts_ref, pts_cur);
    std::cout << "-- Initial succeed? " << succeed << std::endl;

    std::vector<cv::KeyPoint> kps1, kps2;
    cv::KeyPoint::convert(pts_ref, kps1);
    cv::KeyPoint::convert(pts_cur, kps2);
    std::vector<cv::DMatch> matches;

    for(int i=0; i<kps1.size();i++)
    {
        matches.push_back(cv::DMatch(i,i,0));
    }

    cv::Mat match_img;
    cv::drawMatches(ref_img, kps1, cur_img, kps2, matches, match_img);
    cv::imshow("KeyPoints detectByImage", match_img);
    cv::waitKey(0);

    return 0;
}