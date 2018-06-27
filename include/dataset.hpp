#ifndef _SSVO_DATASET_HPP_
#define _SSVO_DATASET_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

namespace ssvo {

class TUMDataReader {
public:
    TUMDataReader(const std::string &dataset_path, const std::string &association_file, const bool with_ground_truth = false):
        dataset_path_(dataset_path), association_file_(association_file)
    {
        size_t found = dataset_path.find_last_of("/\\");
        if(found + 1 != dataset_path.size())
            dataset_path_ += dataset_path.substr(found, 1);

        std::ifstream file_stream;
        file_stream.open(association_file_.c_str());
        while (!file_stream.eof()) {
            std::string s;
            getline(file_stream, s);
            if (!s.empty()) {
                std::stringstream ss;
                ss << s;
                double time;
                std::string rgb, depth;
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
                    groundtruth_data_.push_back(ground_truth);
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
        ground_truth = groundtruth_data_[index];
        return true;
    }

public:
    size_t N;
    std::string dataset_path_;
    std::string association_file_;
    std::vector<double> timestamps_;
    std::vector<std::string> rgb_images_;
    std::vector<std::string> depth_images_;
    std::vector<std::vector<double> > groundtruth_data_;
};

class EuRocDataReader {
public:

    struct Image{
        double timestamp;
        std::string path;
    };

    struct IMUData{
        double timestamp;
        std::array<double, 3> gyro; //! w_RS_S_[x,y,z]
        std::array<double, 3> acc;  //! a_RS_S_[x,y,z]
    };

    struct GroundTruthData{
        double timestamp;
        std::array<double, 3> p; //! p_RS_R_[x,y,z]
        std::array<double, 4> q; //! q_RS_[w,x,y,z]
        std::array<double, 3> v; //! v_RS_R_[x,y,z]
        std::array<double, 3> w; //! b_w_RS_S_[x,y,z]
        std::array<double, 3> a; //! b_a_RS_S_[x,y,z]
    };

    EuRocDataReader(const std::string dataset_path, const bool info = false)
    {
        std::string root_path = dataset_path;
        size_t found = root_path.find_last_of("/\\");
        if(found > root_path.size())
        {
            std::cerr << "EuRocDataReader::load, Please check the path: " + dataset_path << std::endl;
            throw;
        }

        std::string divide_str = root_path.substr(found, 1);
        if (found + 1 != root_path.size())
            root_path += divide_str;

        int code = 0;

        //! for image0
        std::string cam_path0 = root_path + "cam0" + divide_str;
        std::string image_path0 = cam_path0 + "data" + divide_str;
        code = loadCameraData(cam_path0, image_path0, left_);
        if(code != 0)
        {
            std::cerr << "EuRocDataReader, Load cam0 with error " << code <<  ". Please check the cam0 path: " + cam_path0 << std::endl;
            throw;
        }

        //! for image1
        std::string cam_path1 = root_path + "cam1" + divide_str;
        std::string image_path1 = cam_path1 + "data" + divide_str;
        code = loadCameraData(cam_path1, image_path1, right_);
        if(code != 0)
        {
            std::cerr << "EuRocDataReader, Load cam1 with error " << code <<  ". Please check the cam1 path: " + cam_path1 << std::endl;
            throw;
        }

        //! for IMU
        std::string imu_path = root_path + "imu0" + divide_str;
        code = loadIMUData(imu_path, imu_);
        if(code != 0)
        {
            std::cerr << "EuRocDataReader, Load imu with error " << code <<  ". Please check the imu path: " + imu_path << std::endl;
            throw;
        }

        //! for groundtruth
        std::string groundtruth_path = root_path + "state_groundtruth_estimate0" + divide_str;
        code = loadGroundtruthData(groundtruth_path, groundtruth_);
        if(code != 0)
        {
            std::cerr << "EuRocDataReader, Load groundtruth with error " << code <<  ". Please check the groundtruth path: " + groundtruth_path << std::endl;
            throw;
        }

    }

    const size_t leftImageSize() const { return left_.size(); }

    const size_t rightImageSize() const { return right_.size(); }

    const size_t imuSize() const { return imu_.size(); }

    const size_t groundtruthSize() const { return groundtruth_.size(); }

    const Image& leftImage(size_t idx) const { return left_.at(idx); }

    const Image& rightImage(size_t idx) const { return right_.at(idx); }

    const IMUData& imu(size_t idx) const { return imu_.at(idx); }

    const GroundTruthData& groundtruth(size_t idx) const { return groundtruth_.at(idx); }

	bool getGroundtruthAtTime(double time, GroundTruthData &data, const double dt = 1e-3) const {
		for (size_t i = 0; i < groundtruth_.size(); i++)
		{
			if (std::abs(groundtruth_[i].timestamp - time) > dt)
				continue;

			data = groundtruth_[i];
			return true;
		}

		return false;
	}

private:

    int loadCameraData(std::string root_path, std::string data_path, std::vector<Image> &images)
    {
        std::string camera_csv = root_path + "data.csv";

        std::ifstream file_stream(camera_csv, std::ifstream::in);
        if (!file_stream.is_open())
            return -1;

        //! image path
        std::string buffer;
        getline(file_stream, buffer);
        while (getline(file_stream, buffer))
        {
            size_t found = buffer.find_last_not_of(" \t\n\r");
            if (found != std::string::npos)
                buffer.erase(found + 1);
            else
                break;

            std::istringstream string_stream(buffer);
            std::string time_buffer;
            std::string name_buffer;
            getline(string_stream, time_buffer, ',');
            time_t time;
            std::istringstream(time_buffer) >> time;
            getline(string_stream, name_buffer);

            Image image = { time * 1e-9, data_path + name_buffer};
            images.push_back(image);
        }
        file_stream.close();

        //! config file
        std::string camera_yaml = root_path + "sensor.yaml";


        return 0;
    }

    int loadIMUData(std::string path, std::vector<IMUData> &imu)
    {
        //! csv
        std::string imu_csv = path + "data.csv";
        std::ifstream file_stream(imu_csv, std::ifstream::in);
        if (!file_stream.is_open())
            return -1;

        std::string buffer;
        getline(file_stream, buffer);
        while (getline(file_stream, buffer))
        {
            size_t found = buffer.find_last_not_of(" \t\n\r");
            if (found != std::string::npos)
                buffer.erase(found + 1);
            else
                break;

            std::istringstream string_stream(buffer);
            std::string time_buffer;
            std::string imu_buffer;
            getline(string_stream, time_buffer, ',');
            time_t time;
            std::istringstream(time_buffer) >> time;
            getline(string_stream, imu_buffer);

            IMUData data;
            data.timestamp = time * 1e-9;
            char dot;
            std::istringstream(imu_buffer)
                    >> data.gyro[0] >> dot
                    >> data.gyro[1] >> dot
                    >> data.gyro[2] >> dot
                    >> data.acc[0] >> dot
                    >> data.acc[1] >> dot
                    >> data.acc[2];

            assert(dot == ',');

            imu.push_back(data);
        }
        file_stream.close();

        return 0;
    }

    int loadGroundtruthData(std::string path, std::vector<GroundTruthData> &groundtruth)
    {
        std::string ground_truth_csv = path + "data.csv";
        std::ifstream file_stream(ground_truth_csv, std::ifstream::in);
        if (!file_stream.is_open())
            return -1;

        std::string buffer;
        getline(file_stream, buffer);
        while (getline(file_stream, buffer))
        {
            size_t found = buffer.find_last_not_of(" \t\n\r");
            if (found != std::string::npos)
                buffer.erase(found + 1);
            else
                break;

            std::istringstream string_stream(buffer);
            std::string time_buffer;
            std::string pose_buffer;
            getline(string_stream, time_buffer, ',');
            time_t time;
            std::istringstream(time_buffer) >> time;
            getline(string_stream, pose_buffer);

            GroundTruthData data;
            data.timestamp = time * 1e-9;
            char dot;
            std::istringstream(pose_buffer)
                    >> data.p[0] >> dot
                    >> data.p[1] >> dot
                    >> data.p[2] >> dot
                    >> data.q[0] >> dot
                    >> data.q[1] >> dot
                    >> data.q[2] >> dot
                    >> data.q[3] >> dot
                    >> data.v[0] >> dot
                    >> data.v[1] >> dot
                    >> data.v[2] >> dot
                    >> data.w[0] >> dot
                    >> data.w[1] >> dot
                    >> data.w[2] >> dot
                    >> data.a[0] >> dot
                    >> data.a[1] >> dot
                    >> data.a[2];

            assert(dot == ',');

            groundtruth.push_back(data);
        }
        file_stream.close();

        return 0;
    }

private:

    std::vector<Image> left_;
    std::vector<Image> right_;
    std::vector<IMUData> imu_;
    std::vector<GroundTruthData> groundtruth_;
};

}

#endif //_SSVO_DATASET_HPP_
