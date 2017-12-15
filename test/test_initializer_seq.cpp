#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "config.hpp"
#include "optimizer.hpp"
#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

using std::string;
using namespace ssvo;

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

void evalueErrors(KeyFrame::Ptr kf1, KeyFrame::Ptr kf2, double& error)
{
    std::vector<Feature::Ptr> fts1 = kf1->getFeatures();
    double residuals[2] = {0,0};
    Matrix3d R = kf2->Tcw().rotationMatrix();
    Vector3d t = kf2->Tcw().translation();
    for(Feature::Ptr ft1:fts1)
    {
        MapPoint::Ptr mpt = ft1->mpt;

        if(mpt == nullptr)
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        Vector3d p1 = mpt->pose();
        Vector3d p2 = R*p1 + t;

        double predicted_x1 = p1[0] / p1[2];
        double predicted_y1 = p1[1] / p1[2];
        double dx1 = predicted_x1 - ft1->ft[0]/ft1->ft[2];
        double dy1 = predicted_y1 - ft1->ft[1]/ft1->ft[2];
        residuals[0] += dx1*dx1 + dy1*dy1;

        double predicted_x2 = p2[0] / p2[2];
        double predicted_y2 = p2[1] / p2[2];
        double dx2 = predicted_x2 - ft2->ft[0]/ft2->ft[2];
        double dy2 = predicted_y2 - ft2->ft[1]/ft2->ft[2];
        residuals[1] += dx2*dx2 + dy2*dy2;
    }

    error = 0.5*(residuals[0] + residuals[1]);
}

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 3) {
        std::cout << "Usge: ./test_initializer path_to_sequence configflie" << std::endl;
        return -1;
    }

    Config::FileName = std::string(argv[2]);
    int fps = Config::cameraFps();
    int width = Config::imageWidth();
    int height = Config::imageHeight();
    int levels = Config::imageTopLevel();
    int image_border = 8;
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();

    std::string dir_name = argv[1];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);

    cv::Mat K = Config::cameraIntrinsic();
    cv::Mat DistCoef = Config::cameraDistCoef();

    Camera::Ptr camera = Camera::create(Config::imageWidth(), Config::imageHeight(), K, DistCoef);
    FastDetector::Ptr detector = FastDetector::create(width, height, image_border, levels+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    Initializer::Ptr initializer = Initializer::create(detector);

    int initial = 0;
    std::vector<Corner> corners;
    std::vector<Corner> old_corners;
    Frame::Ptr frame_ref, frame_cur;
    for(std::vector<std::string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if(img.empty()) throw std::runtime_error("Could not open image: " + *i);

        std::cout << "Load Image: " << *i << std::endl;
        cv::Mat cur_img;
        cv::cvtColor(img, cur_img, cv::COLOR_RGB2GRAY);

        if(initial == 0)
        {
            frame_ref = Frame::create(cur_img, 0, camera);
            if(initializer->addFirstImage(frame_ref) == SUCCESS)
                initial = 1;
        }
        else if(initial == 1)
        {
            frame_cur = Frame::create(cur_img, 1, camera);
            InitResult result = initializer->addSecondImage(frame_cur);
            if(result == RESET) {
                initial = 0;
                //continue;
            }
            else if(result == SUCCESS)
                break;

            cv::Mat klt_img;
            initializer->drowOpticalFlow(img, klt_img);
            cv::imshow("KLTracking", klt_img);
        }

        cv::waitKey(fps);
    }


    ssvo::Map::Ptr map = ssvo::Map::create();
    initializer->createInitalMap(map, 1.0);

    std::vector<KeyFrame::Ptr> kfs = map->getAllKeyFramesOrderedByID();
    LOG_ASSERT(kfs.size() == 2) << "Error number of keyframes in map after initailizer: " << kfs.size();
    LOG_ASSERT(kfs[0]->id_ == 0 && kfs[1]->id_ == 1) << "Error id of keyframe: " << kfs[0]->id_ << ", " << kfs[0]->id_;

    double error = 0;
    evalueErrors(kfs[0], kfs[1], error);
    LOG(INFO) <<"Pose:\n" << kfs[1]->pose().matrix();
    LOG(INFO) << "Error before BA: " << error;

    Optimizer optimizer;
    optimizer.twoViewBundleAdjustment(kfs[0], kfs[1], nullptr);
    optimizer.report(true);

    LOG(INFO) <<"Pose:\n" << kfs[1]->pose().matrix();
    evalueErrors(kfs[0], kfs[1], error);
    LOG(INFO) << "Error after BA: " << error;


    cv::waitKey(0);

    return 0;
}