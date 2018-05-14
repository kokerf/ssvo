#include <fstream>
#include <sstream>
#include "global.hpp"
#include "utils.hpp"
#include "camera.hpp"

//! please use the TUM mono dataset
//! https://vision.in.tum.de/data/datasets/mono-dataset

class UndistorterFOV
{
public:

    UndistorterFOV(ssvo::AbstractCamera::Ptr cam) : cam_(cam) { getundistortMap();}

    void getundistortMap()
    {
        remapX_ = cv::Mat::zeros(cam_->height(), cam_->width(), CV_32FC1);
        remapY_ = cv::Mat::zeros(cam_->height(), cam_->width(), CV_32FC1);

        for (int y = 0; y < cam_->height(); ++y)
            for (int x = 0; x < cam_->width(); ++x)
            {
                double x_norm = (x - cam_->cx()) / cam_->fx();
                double y_norm = (y - cam_->cy()) / cam_->fy();
                Vector2d px_dist = cam_->project(x_norm, y_norm);
                remapX_.at<float>(y, x) = static_cast<float>(px_dist[0]);
                remapY_.at<float>(y, x) = static_cast<float>(px_dist[1]);
            }
    }

    void undistort(const cv::Mat &img_dist, cv::Mat &img_udist)
    {
        LOG_ASSERT(img_dist.cols == cam_->width() && img_dist.rows == cam_->height()) << "Error input image size";
        LOG_ASSERT(img_dist.type() == CV_8UC1) << "Error input image type";
        img_udist = cv::Mat::zeros(cam_->height(), cam_->width(), CV_8UC1);

        for (int y = 0; y < cam_->height(); ++y)
            for (int x = 0; x < cam_->width(); ++x)
            {
                float x_d = remapX_.at<float>(y, x);
                float y_d = remapY_.at<float>(y, x);

                if (!cam_->isInFrame(Vector2i(x_d, y_d)))
                    img_udist.at<float>(y, x) = 0;

                img_udist.at<uchar>(y,x) = 0.5 + ssvo::utils::interpolateMat<uchar, float>(img_dist, x_d, y_d);
            }

    }

public:

    ssvo::AbstractCamera::Ptr cam_;
    cv::Mat remapX_;
    cv::Mat remapY_;

};

int main(int argc, char *argv[])
{
    LOG_ASSERT(argc == 2) << "run as: ./test_camera_model dataset_dir";
    std::string dataset = argv[1];
    std::ifstream ifstream_config(dataset+"camera.txt");
    std::ifstream ifstream_times(dataset+"times.txt");
    if(!ifstream_config.good())
    {
        std::cout << "Failed to read camera calibration file: " <<  dataset+"camera.txt" << std::endl;
        return -1;
    }

    if(!ifstream_times.good())
    {
        std::cout << "Failed to read image timestamp file: " <<  dataset+"times.txt" << std::endl;
        return -1;
    }

    std::string l1, l2;
    std::getline(ifstream_config, l1);
    std::getline(ifstream_config, l2);

    int width, height;
    float fx, fy, cx, cy, s;
    if(std::sscanf(l1.c_str(), "%f %f %f %f %f", &fx, &fy, &cx, &cy, &s) == 5 &&
            std::sscanf(l2.c_str(), "%d %d", &width, &height) == 2)
    {
        std::cout << "Input resolution: " << width << " " << height << std::endl;
        std::cout << "Input Calibration (fx fy cx cy): "
                  << width * fx << " " << height * fy << " " << width * cx << " " << height * cy << std::endl;
    }
    else
    {
        std::cout << "Failed to read camera calibration parameters: " <<  dataset+"camera.txt" << std::endl;
        return -1;
    }

    ssvo::AtanCamera::Ptr atan_cam = ssvo::AtanCamera::create(width, height, fx, fy, cx, cy, s);

    UndistorterFOV undistorter(std::static_pointer_cast<ssvo::AbstractCamera>(atan_cam));

    bool is_undistort = false;
    bool is_autoplay = false;
    std::string timestamp_line;
    while(std::getline(ifstream_times, timestamp_line))
    {
        std::string id;
        std::istringstream istream_line(timestamp_line);
        istream_line >> id;
        cv::Mat image_input = cv::imread(dataset + "images/" + id + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
        LOG_ASSERT(!image_input.empty()) << "Error in loading image: " + dataset + "images/"+ id + ".jpg";
        cv::Mat image_undistorted;
        undistorter.undistort(image_input, image_undistorted);;

        while(true)
        {
            cv::Mat show = is_undistort ? image_undistorted : image_input;
            cv::imshow("image show", show);

            char k;
            if(is_autoplay) k = cv::waitKey(1);
            else k = cv::waitKey(0);

            if(k == ' ') break;
            if(k == 'r' || k == 'R') is_undistort = !is_undistort;
            if(k == 'a' || k == 'A') is_autoplay = !is_autoplay;
            if(is_autoplay) break;
        }
    }

    std::cout << "End!!!" << std::endl;

    cv::waitKey(0);

    return 0;
}