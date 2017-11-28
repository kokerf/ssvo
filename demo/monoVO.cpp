#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <memory>
#include "system.hpp"

using namespace ssvo;

class EuRocDataReader{
public:
    EuRocDataReader(const std::string image_path, const std::string csv_file)
    {
        image_path_ = image_path;
        size_t found = image_path_.find_last_of("/\\");
        if(found+1 != image_path_.size())
            image_path_ += image_path_.substr(found,1);

        input_stream_.open(csv_file, std::ifstream::in);
        //LOG_ASSERT(input_stream_.is_open()) << "Error in open file: " << csv_file;
        std::string buffer;
        getline(input_stream_, buffer);
    }

    ~EuRocDataReader()
    {
        input_stream_.close();
    }

    bool getOneItem(std::string& image_name, double& timestamp)
    {
        std::string buffer;
        if(!getline(input_stream_, buffer))
            return false;

        std::size_t found = buffer.find_last_not_of(" \t\n\r");
        if (found!=std::string::npos)
            buffer.erase(found+1);
        else
            return false;

        std::istringstream string_stream(buffer);
        std::string time_buffer;
        std::string name_buffer;
        getline(string_stream, time_buffer, ',');
        time_t time;
        std::istringstream(time_buffer) >> time;
        timestamp = time/1000000000.0;
        getline(string_stream, name_buffer);
        image_name = image_path_ + name_buffer;

        return true;
    }

private:
    std::string image_path_;
    std::ifstream input_stream_;
};

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 4) << "\n Usage : ./monoVO config_file dataset_image_path  dataset_csv_file";

    System vo(argv[1]);
    EuRocDataReader data(argv[2], argv[3]);

    std::string image_name;
    double timestamp;
    while(data.getOneItem(image_name, timestamp))
    {
        LOG(INFO) << "Image: " << image_name << ", time: " << std::fixed <<std::setprecision(7)<< timestamp << std::endl;
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
        vo.process(image, timestamp);

        //cv::imshow("Image Show", image);
        cv::waitKey(40);
    }

    return 0;
}