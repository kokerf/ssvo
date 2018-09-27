#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "feature_detector.hpp"
#include "camera.hpp"
#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

using namespace cv;
using namespace ssvo;

#ifdef WIN32
void loadImages(const string &file_directory, std::vector<string> &image_filenames)
{
    _finddata_t file;
    intptr_t fileHandle;
    std::string filename = file_directory + "\\*.jpg";
    fileHandle = _findfirst(filename.c_str(), &file);
    if(fileHandle == -1L)
    {
        filename = file_directory + "\\*.png";
        fileHandle = _findfirst(filename.c_str(), &file);
    }

    if(fileHandle  != -1L)
    {
        do {
            std::string image_name = file_directory + "\\" + file.name;
            image_filenames.push_back(image_name);
            std::cout << image_name << std::endl;
        } while (_findnext(fileHandle, &file) == 0);
    }

    _findclose(fileHandle);
    std::sort(image_filenames.begin(), image_filenames.end());
}
#else
void loadImages(const std::string &strFileDirectory, std::vector<string> &vstrImageFilenames)
{
    DIR* dir = opendir(strFileDirectory.c_str());
    dirent* p = NULL;
    while((p = readdir(dir)) != NULL)
    {
        if(p->d_name[0] != '.')
        {
            std::string imageFilename = strFileDirectory + string(p->d_name);
            vstrImageFilenames.push_back(imageFilename);
            //cout << imageFilename << endl;
        }
    }

    closedir(dir);
    std::sort(vstrImageFilenames.begin(),vstrImageFilenames.end());
}
#endif

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size);

int main(int argc, char const *argv[])
{
    if(argc != 4)
    {
        std::cout << "Usge: ./test_feature_detector calib_file config_file path_to_sequence" << std::endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);

    std::string dir_name = argv[3];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);
    LOG_ASSERT(!img_file_names.empty()) << "No images load from " << dir_name;

    ssvo::PinholeCamera::Ptr pinhole_cam = ssvo::PinholeCamera::create(argv[1]);
    Config::file_name_ = std::string(argv[2]);
    int width = pinhole_cam->width();
    int height = pinhole_cam->height();
    int nlevels = Config::imageNLevels();
    double scale_factor = Config::imageScaleFactor();

    int image_border = 8;
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    double fast_min_eigen = Config::fastMinEigen();

    cv::Mat image = cv::imread(img_file_names[0], CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Could not open image: " + img_file_names[0]);

    std::vector<cv::Mat> image_pyramid;
    computePyramid(image, image_pyramid, 2, 4, cv::Size(40, 40));

    Corners new_corners, old_corners;
    FastDetector::Ptr fast_detector = FastDetector::create(width, height, image_border, nlevels, scale_factor, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    LOG(WARNING) << "=== This is a FAST corner detector demo ===";
    const int n_trials = 1000;
    double time_accumulator = 0;
    for(int i = 0; i < n_trials; ++i)
    {
        double t = (double)cv::getTickCount();
        fast_detector->detect(image_pyramid, new_corners, old_corners, 100, fast_min_eigen);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
        LOG_EVERY_N(WARNING, n_trials/20) << " i: " << i << ", new_corners: " << new_corners.size();
    }
    LOG(WARNING) << " took " <<  time_accumulator/((double)n_trials)*1000.0
              << " ms (average over " << n_trials << " trials)." << std::endl;
    cv::Mat kps_img;
    std::vector<cv::KeyPoint> keypoints;
    std::for_each(new_corners.begin(), new_corners.end(), [&](Corner corner){
      cv::KeyPoint kp(corner.x, corner.y, 0);
      keypoints.push_back(kp);
    });
    cv::drawKeypoints(image, keypoints, kps_img);

    fast_detector->drawGrid(kps_img, kps_img);
    cv::imshow("KeyPoints detectByImage", kps_img);
    cv::waitKey(0);

    old_corners = new_corners;
    old_corners.resize(old_corners.size()/2);
    time_accumulator = 0;
    for(int i = 0; i < n_trials; ++i)
    {
        double t = (double)cv::getTickCount();
        fast_detector->detect(image_pyramid, new_corners, old_corners, 100, fast_min_eigen);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
        LOG_EVERY_N(WARNING, n_trials/20) << " i: " << i << ", new_corners: " << new_corners.size();
    }
    LOG(WARNING) << " took " <<  time_accumulator/((double)n_trials)*1000.0
                 << " ms (average over " << n_trials << " trials)." << std::endl;

    cv::Mat kps_img1;
    std::vector<cv::KeyPoint> keypoints1;
    std::for_each(new_corners.begin(), new_corners.end(), [&](Corner corner){
        cv::KeyPoint kp(corner.x, corner.y, 0);
        keypoints1.push_back(kp);
    });
    cv::drawKeypoints(image, keypoints1, kps_img1);

    fast_detector->drawGrid(kps_img1, kps_img1);
    cv::imshow("KeyPoints detectByImage1", kps_img1);
    cv::waitKey(0);

    LOG(INFO) << "=== Test Adaptive Feature detector ===";
    Ptr<ORB> orb = ORB::create();
    cv::Mat descriptor;
    old_corners.clear();
    time_accumulator = 0;
    int count = 0;
    for(std::vector<std::string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i) {
        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if (img.empty()) throw std::runtime_error("Could not open image: " + *i);

        LOG(WARNING) << "Load Image: " << *i << std::endl;
        cv::Mat cur_img = img.clone();
        if(img.channels()!=1)
            cv::cvtColor(cur_img, cur_img, cv::COLOR_RGB2GRAY);
        std::vector<cv::Mat> image_pyramid;
        computePyramid(cur_img, image_pyramid, 2, 4, cv::Size(40, 40));

        double t0 = (double)cv::getTickCount();
        fast_detector->detect(image_pyramid, new_corners, old_corners, 100, fast_min_eigen);


        cv::Mat kps_img;
        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(new_corners.size());
        std::for_each(new_corners.begin(), new_corners.end(), [&](Corner corner){
          cv::KeyPoint kp(corner.x, corner.y, 0, -1, corner.score, corner.level);
          keypoints.push_back(kp);
        });
        double t1 = (double)cv::getTickCount();
        orb->compute(cur_img, keypoints, descriptor);
        double delta_t = (t1 - t0) / cv::getTickFrequency();
        time_accumulator += delta_t;
        count++;
        LOG(WARNING) << "Time: " << delta_t << ", "
                     << (cv::getTickCount() - t1) / cv::getTickFrequency()<< ", corners: " << new_corners.size();

        cv::drawKeypoints(cur_img, keypoints, kps_img);

        fast_detector->drawGrid(kps_img, kps_img);
        cv::imshow("KeyPoints", kps_img);
        cv::waitKey(20);
    }
    LOG(WARNING) << " took " <<  time_accumulator/((double)count)*1000.0
                 << " ms (average over " << count << " images)." << std::endl;

    return 0;
}

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size)
{
    LOG_ASSERT(scale_factor > 1.0);
    LOG_ASSERT(!image.empty());

    image_pyramid.resize(level + 1);

    image_pyramid[0] = image.clone();
    for(int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i - 1].cols / scale_factor), round(image_pyramid[i - 1].rows / scale_factor));

        if(size.height < min_size.height || size.width < min_size.width)
        {
            image_pyramid.resize(level);
            return level-1;
        }

        cv::resize(image_pyramid[i - 1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return level;
}