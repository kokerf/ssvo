#include "imu.hpp"
#include "global.hpp"

namespace ssvo{

IMUProcessor::IMUProcessor() :
        is_initialed_(false), gyro_bais_(0.0, 0.0, 0.0), acc_bais_(0.0, 0.0, 0.0), gravity_(0.0, 0.0, -9.81)
{}

void IMUProcessor::addIMUData(const ssvo::IMUData &data)
{
    std::unique_lock<std::mutex> lock(mutex_imu_);
    imu_buffer_.push_back(data);
}

bool IMUProcessor::isInitialized() const
{
    return is_initialed_;
}

bool IMUProcessor::initializeGravityAndBias()
{
    std::unique_lock<std::mutex> lock(mutex_imu_);

    Vector3d gyro_sum = Vector3d::Zero();
    Vector3d acc_sum = Vector3d::Zero();

    const size_t N = imu_buffer_.size();
    if (N < 400)
        return false;

    while (!imu_buffer_.empty())
    {
        gyro_sum.noalias() += imu_buffer_.front().gyro;
        acc_sum.noalias() += imu_buffer_.front().acc;
        imu_buffer_.pop_front();
    }

    Vector3d gravity_imu = acc_sum / N;

    gyro_bais_ = gyro_sum / N;
    gravity_ = Vector3d(0, 0, -gravity_imu.norm());
    is_initialed_ = true;

    return true;
}

void IMUProcessor::getIMUData(const double timestamp, std::vector<IMUData> &imu_data)
{
    std::unique_lock<std::mutex> lock(mutex_imu_);
    imu_data.clear();

    while(!imu_buffer_.empty())
    {
        const IMUData data = imu_buffer_.front();
        if(data.timestamp > timestamp)
            break;

        imu_data.push_back(data);
        imu_buffer_.pop_front();
    }
}

}

