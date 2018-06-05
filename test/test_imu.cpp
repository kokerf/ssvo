#include "dataset.hpp"
#include "camera.hpp"
#include "imu.hpp"


using namespace ssvo;
using std::string;

int main(int argc, char *argv[])
{
    assert(argc == 3);

    PinholeCamera::Ptr cam = PinholeCamera::create(argv[1]);

    EuRocDataReader dataset(argv[2]);

    IMUProcessor::Ptr imu_processor = IMUProcessor::create();

    size_t imu_idx = 0;
    size_t ground_truth_idx = 0;
    EuRocDataReader::Image image_cur = dataset.leftImage(0);
    EuRocDataReader::Image image_last = dataset.leftImage(0);

    EuRocDataReader::GroundTruthData ground_truth_last;
    EuRocDataReader::GroundTruthData ground_truth_cur;

    while (1)
    {
        ground_truth_cur = dataset.groundtruth(ground_truth_idx);
        if(ground_truth_cur.timestamp >= image_cur.timestamp)
            break;

        ground_truth_idx++;
    }

    bool aligned_last = false;
    bool aligned_cur = false;
    for (size_t i = 0; i < dataset.leftImageSize(); ++i)
    {
        //! load image
        image_last = image_cur;
        image_cur = dataset.leftImage(i);

        //! load imu
        while (imu_idx < dataset.imuSize())
        {
            EuRocDataReader::IMUData imu_data = dataset.imu(imu_idx);
            if (imu_data.timestamp >= image_cur.timestamp)
                break;

            imu_processor->addIMUData(ssvo::IMUData(imu_data.timestamp,
                                                    Vector3d(imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]),
                                                    Vector3d(imu_data.acc[0], imu_data.acc[1], imu_data.acc[2])));
            imu_idx++;
        }

        //! load groundtruth
        ground_truth_last = ground_truth_cur;
        aligned_last = aligned_cur;
        aligned_cur =false;
        while (ground_truth_idx < dataset.groundtruthSize())
        {
            ground_truth_cur = dataset.groundtruth(ground_truth_idx);
            if (std::abs(ground_truth_cur.timestamp - image_cur.timestamp) < 0.002)
            {
                aligned_cur = true;
                break;
            }
            else if(ground_truth_cur.timestamp >= image_cur.timestamp)
                break;

            ground_truth_idx++;
        }

        //! process
        if(!imu_processor->isInitialized())
            imu_processor->initializeGravityAndBias();
        else
        {
            std::vector<IMUData> imu_data;
            imu_processor->getIMUData(image_cur.timestamp, imu_data);
            if(!imu_data.empty())
            {
                Vector3d mean_angle_vel(0.0, 0.0, 0.0);
                for (const IMUData &data : imu_data)
                    mean_angle_vel += (data.gyro - imu_processor->gyroBais());

                mean_angle_vel *= 1.0 / imu_data.size();

                Vector3d cam_angle_vel =  mean_angle_vel;//cam->T_CB().rotationMatrix() * mean_angle_vel;
                const double angle = (image_cur.timestamp - image_last.timestamp) * cam_angle_vel.norm();
                cam_angle_vel.normalize();

                Matrix3d R = AngleAxisd(angle, cam_angle_vel).toRotationMatrix();
                std::cout << "R:\n" << R << std::endl;
                std::cout << "Angle:\n" << angle << ", " << cam_angle_vel.transpose() << std::endl;
            }

            if(aligned_cur && aligned_last)
            {
                Quaterniond q_last(ground_truth_last.q[0], ground_truth_last.q[1], ground_truth_last.q[2], ground_truth_last.q[3]);
                Quaterniond q_cur(ground_truth_cur.q[0], ground_truth_cur.q[1], ground_truth_cur.q[2], ground_truth_cur.q[3]);

                Quaterniond q_last_from_cur = q_last.inverse() * q_cur;
                AngleAxisd angle_last_from_cur = AngleAxisd(q_last_from_cur);
                std::cout << "ground truth: " << angle_last_from_cur.angle() << ", " << angle_last_from_cur.axis().transpose() << std::endl;
            }

        }
    }
}