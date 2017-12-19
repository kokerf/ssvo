#ifndef _SSVO_DATASET_HPP_
#define _SSVO_DATASET_HPP_

#include <iostream>
#include <fstream>
#include <vector>

namespace ssvo {

class TUMDataReader {
public:
    TUMDataReader(const string &dataset_path, const string &association_file, const bool with_ground_truth = false):
        dataset_path_(dataset_path), association_file_(association_file)
    {
        size_t found = dataset_path.find_last_of("/\\");
        if(found + 1 != dataset_path.size())
            dataset_path_ += dataset_path.substr(found, 1);

        std::ifstream file_stream;
        file_stream.open(association_file_.c_str());
        while (!file_stream.eof()) {
            string s;
            getline(file_stream, s);
            if (!s.empty()) {
                std::stringstream ss;
                ss << s;
                double time;
                string rgb, depth;
                ss >> time;
                timestamps_.push_back(time);
                ss >> rgb;
                rgb_images_.push_back(dataset_path_ + rgb);
                ss >> time;
                ss >> depth;
                depth_images_.push_back(dataset_path_ + depth);
                ss >> time;
                if(with_ground_truth)
                {
                    std::vector<double> ground_truth(7);
                    for(int i = 0; i < 7; ++i)
                       ss >> ground_truth[i];
                    ground_truth_.push_back(ground_truth);
                }
            }
        }
        file_stream.close();

        N = timestamps_.size();
        if(N == 0)
            std::cerr << "No item read! Please check association file: " << association_file << std::endl;
        else
            std::cerr << "Avaiable image items in dataset: " << N << std::endl;

    }

    bool readItemByIndex(size_t index, std::string &rgb_image, std::string &depth_image, double &timestamp) const
    {
        if(index >= N)
        {
            std::cerr << " Index(" << index << ") is out of scape, max should be (0~" << N - 1 << ")";
            return false;
        }
        rgb_image = rgb_images_[index];
        depth_image = depth_images_[index];
        timestamp = timestamps_[index];
        return true;
    }

    bool readItemWithGroundTruth(size_t index, std::string &rgb_image, std::string &depth_image, double &timestamp, std::vector<double> &ground_truth) const
    {
        if(!readItemByIndex(index, rgb_image, depth_image, timestamp))
            return false;
        ground_truth = ground_truth_[index];
        return true;
    }

public:
    size_t N;
    std::string dataset_path_;
    std::string association_file_;
    std::vector<double> timestamps_;
    std::vector<std::string> rgb_images_;
    std::vector<std::string> depth_images_;
    std::vector<std::vector<double> > ground_truth_;
};

class EuRocDataReader{
public:
    EuRocDataReader(const std::string image_path, const std::string csv_file)
    {
        std::string path = image_path;
        size_t found = path.find_last_of("/\\");
        if(found+1 != path.size())
            path += path.substr(found,1);

        std::ifstream file_stream;
        file_stream.open(csv_file, std::ifstream::in);

        std::string buffer;
        getline(file_stream, buffer);
        while(getline(file_stream, buffer))
        {
            size_t found = buffer.find_last_not_of(" \t\n\r");
            if(found!=std::string::npos)
                buffer.erase(found+1);
            else
                break;

            std::istringstream string_stream(buffer);
            std::string time_buffer;
            std::string name_buffer;
            getline(string_stream, time_buffer, ',');
            time_t time;
            std::istringstream(time_buffer) >> time;
            double timestamp = time/1000000000.0;
            getline(string_stream, name_buffer);
            std::string image_path = path+name_buffer;
            timestamps_.push_back(timestamp);
            images_.push_back(image_path);

        }
        file_stream.close();
        N = timestamps_.size();

        if(N == 0)
            std::cerr << "No item read! Please check csv file: " << csv_file << std::endl;
        else
            std::cerr << "Avaiable image items in dataset: " << N << std::endl;
    }

    bool readItemByIndex(size_t index, std::string &rgb_image, double &timestamp) const
    {
        if(index >= N)
        {
            std::cerr << " Index(" << index << ") is out of scape, max should be (0~" << N - 1 << ")";
            return false;
        }
        rgb_image = images_[index];
        timestamp = timestamps_[index];
        return true;
    }

public:
    size_t N;
    std::vector<double> timestamps_;
    std::vector<std::string> images_;
};

}

#endif //_SSVO_DATASET_HPP_
