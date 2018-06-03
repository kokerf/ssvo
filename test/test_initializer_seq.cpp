#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "local_mapping.hpp"
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
    if(fileHandle == -1L)
    {
        filename = file_directory + "\\*.png";
        fileHandle = _findfirst(filename.c_str(), &file);
    }

    if(fileHandle != -1L)
    {
        do {
            std::string image_name = file_directory + "\\" + file.name;
            image_filenames.push_back(image_name);
            std::cout << image_name << std::endl;
        } while(_findnext(fileHandle, &file) == 0);
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
    std::vector<Feature::Ptr> fts1;
    kf1->getFeatures(fts1);
    double residuals[2] = {0,0};
    Matrix3d R = kf2->Tcw().rotationMatrix();
    Vector3d t = kf2->Tcw().translation();
    for(Feature::Ptr ft1:fts1)
    {
        MapPoint::Ptr mpt = ft1->mpt_;

        if(mpt == nullptr)
            continue;

        Feature::Ptr ft2 = mpt->findObservation(kf2);

        Vector3d p1 = mpt->pose();
        Vector3d p2 = R*p1 + t;

        double predicted_x1 = p1[0] / p1[2];
        double predicted_y1 = p1[1] / p1[2];
        double obversed_x1 = ft1->fn_[0]/ft1->fn_[2];
        double obversed_y1 = ft1->fn_[1]/ft1->fn_[2];
        double dx1 = predicted_x1 - obversed_x1;
        double dy1 = predicted_y1 - obversed_y1;
        double res1 = dx1*dx1 + dy1*dy1;
        residuals[0] += res1;

        double predicted_x2 = p2[0] / p2[2];
        double predicted_y2 = p2[1] / p2[2];
        double obversed_x2 = ft2->fn_[0]/ft2->fn_[2];
        double obversed_y2 = ft2->fn_[1]/ft2->fn_[2];
        double dx2 = predicted_x2 - obversed_x2;
        double dy2 = predicted_y2 - obversed_y2;
        double res2 = dx2*dx2 + dy2*dy2;
        residuals[1] += res2;
//        std::cout << "[" << obversed_x1 << ", " << obversed_y1 << "]-[" << predicted_x1 <<", " << predicted_y1 <<"] "
//                  << "[" << obversed_x2 << ", " << obversed_y2 << "]-[" << predicted_x2 <<", " << predicted_y2 <<"] "
//                  << res1 << " " << res2 << " " << 0.5*(res1+res2) << std::endl;
    }

    error = 0.5*(residuals[0] + residuals[1]);
}

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 4) {
        std::cout << "Usge: ./test_initializer calib_file config_flie path_to_sequence" << std::endl;
        return -1;
    }

    AbstractCamera::Ptr camera = std::static_pointer_cast<AbstractCamera>(PinholeCamera::create(argv[1]));
    Config::file_name_ = std::string(argv[2]);
    int fps = camera->fps();
    int width = camera->width();
    int height = camera->height();
    int nlevel = Config::imageNLevel();
    int image_border = 8;
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();

    std::string dir_name = argv[3];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);
    LOG_ASSERT(!img_file_names.empty()) << "Error! No image in directory: " << dir_name;

    cv::Mat K = camera->K();
    cv::Mat DistCoef = camera->D();

    FastDetector::Ptr detector = FastDetector::create(width, height, image_border, nlevel, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    Initializer::Ptr initializer = Initializer::create(detector, true);

    std::vector<Corner> corners;
    std::vector<Corner> old_corners;
    Frame::Ptr frame_cur;
    for(auto i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if(img.empty()) throw std::runtime_error("Could not open image: " + *i);

        std::cout << "Load Image: " << *i << std::endl;
        cv::Mat gray = img.clone();
        if(gray.channels() != 1)
            cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);


        frame_cur = Frame::create(gray, 0, camera);
        Initializer::Result res = initializer->addImage(frame_cur);

        if(res == Initializer::RESET)
            initializer->reset();
        else if(res == Initializer::SUCCESS)
            break;

        cv::Mat klt_img;
        initializer->drowOpticalFlow(klt_img);
        cv::imshow("KLTracking", klt_img);

        cv::waitKey(fps);
    }

    ssvo::LocalMapper::Ptr mapper = ssvo::LocalMapper::create(fps);
    std::vector<Vector3d> points;
    initializer->createInitalMap(1.0);
    mapper->createInitalMap(initializer->getReferenceFrame(), frame_cur);

    KeyFrame::Ptr kf0 = mapper->map_->getKeyFrame(0);
    KeyFrame::Ptr kf1 = mapper->map_->getKeyFrame(1);
    LOG_ASSERT(kf0 != nullptr && kf1 != nullptr) << "Can not find intial keyframes in map!";

    double error = 0;
    evalueErrors(kf0, kf1, error);
    LOG(INFO) <<"Pose:\n" << kf0->pose().matrix();
    LOG(INFO) << "Error before BA: " << error;

    Optimizer::twoViewBundleAdjustment(kf0, kf1, true, true);

    LOG(INFO) <<"Pose:\n" << kf1->pose().matrix();
    evalueErrors(kf0, kf1, error);
    LOG(INFO) << "Error after BA: " << error;

    cv::waitKey(0);

    return 0;
}