#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "config.hpp"
#include "bundle_adjustment.hpp"
#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

using std::string;

#ifdef WIN32
void loadImages(const string &file_directory, std::vector<string> &image_filenames)
{
    _finddata_t file;
    intptr_t fileHandle;
    std::string filename = file_directory + "\\*.jpg";
    fileHandle = _findfirst(filename.c_str(), &file);
    if (fileHandle  != -1L) {
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

void evalueErrors(ssvo::KeyFrame::Ptr kf1, ssvo::KeyFrame::Ptr kf2, double& error)
{
    const int N = kf1->fts_.size();
    double residuals[2] = {0,0};
    Matrix3d R = kf2->pose().rotationMatrix();
    Vector3d t = kf2->pose().translation();
    for(int i = 0; i < N; i++)
    {
        ssvo::Feature::Ptr ft1 = kf1->fts_[i];

        ssvo::MapPoint::Ptr mpt = ft1->mpt;

        if(mpt == nullptr)
            continue;

        ssvo::Feature::Ptr ft2 = mpt->findObservation(kf2);

        Vector3d p1 = mpt->pose();
        Vector3d p2 = R*p1 + t;

        double predicted_x1 = p1[0] / p1[2];
        double predicted_y1 = p1[1] / p1[2];
        double dx1 = predicted_x1 - ft1->ft[0];
        double dy1 = predicted_y1 - ft1->ft[1];
        residuals[0] += dx1*dx1 + dy1*dy1;
        LOG_IF(ERROR, isnan(residuals[0])) << "i: " << i;

        double predicted_x2 = p2[0] / p2[2];
        double predicted_y2 = p2[1] / p2[2];
        double dx2 = predicted_x2 - ft2->ft[0];
        double dy2 = predicted_y2 - ft2->ft[1];
        residuals[1] += dx2*dx2 + dy2*dy2;
    }

    error = 0.5*(residuals[0] + residuals[1]);
}

std::string ssvo::Config::FileName;

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 3) {
        std::cout << "Usge: ./test_initializer path_to_sequence configflie" << std::endl;
        return -1;
    }

    ssvo::Config::FileName = std::string(argv[2]);
    int fps = ssvo::Config::cameraFps();
    int width = ssvo::Config::imageWidth();
    int height = ssvo::Config::imageHeight();
    int levels = ssvo::Config::imageLevels();
    int image_border = ssvo::Config::imageBorder();
    int grid_size = ssvo::Config::gridSize();
    int grid_min_size = ssvo::Config::gridMinSize();
    int fast_max_threshold = ssvo::Config::fastMaxThreshold();
    int fast_min_threshold = ssvo::Config::fastMinThreshold();

    std::string dir_name = argv[1];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);

    cv::Mat K = ssvo::Config::cameraIntrinsic();
    cv::Mat DistCoef = ssvo::Config::cameraDistCoef();

    ssvo::Camera::Ptr camera = ssvo::Camera::create(ssvo::Config::imageWidth(), ssvo::Config::imageHeight(), K, DistCoef);
    ssvo::FastDetector fast(width, height, image_border, levels+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    ssvo::Initializer initializer(&fast);

    int initial = 0;
    std::vector<ssvo::Corner> corners;
    std::vector<ssvo::Corner> old_corners;
    ssvo::Frame::Ptr frame_ref, frame_cur;
    for(std::vector<std::string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if(img.empty()) throw std::runtime_error("Could not open image: " + *i);

        cv::Mat cur_img;
        cv::cvtColor(img, cur_img, cv::COLOR_RGB2GRAY);

        if(initial == 0)
        {
            frame_ref = ssvo::Frame::create(cur_img, 0, camera);
            if(initializer.addFirstImage(frame_ref) == ssvo::SUCCESS)
                initial = 1;
        }
        else if(initial == 1)
        {
            frame_cur = ssvo::Frame::create(cur_img, 1, camera);
            ssvo::InitResult result = initializer.addSecondImage(frame_cur);
            if(result == ssvo::RESET) {
                initial = 0;
                //continue;
            }
            else if(result == ssvo::SUCCESS)
                break;

            cv::Mat klt_img;
            initializer.drowOpticalFlow(img, klt_img);
            cv::imshow("KLTracking", klt_img);
        }

        cv::waitKey(fps);
    }

    ssvo::KeyFrame::Ptr keyframe1 = ssvo::KeyFrame::create(frame_ref);
    ssvo::KeyFrame::Ptr keyframe2 = ssvo::KeyFrame::create(frame_cur);
    keyframe1->updateObservation();
    keyframe2->updateObservation();

    double error = 0;
    evalueErrors(keyframe1, keyframe2, error);
    LOG(INFO) << "Error before BA: " << error;

    ssvo::BA::twoViewBA(keyframe1, keyframe2, nullptr);

    evalueErrors(keyframe1, keyframe2, error);
    LOG(INFO) << "Error after BA: " << error;


    cv::waitKey(0);

    return 0;
}