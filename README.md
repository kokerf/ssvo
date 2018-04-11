# ssvo

Semi-direct sparse odometry

video of running in [rpg_urban dataset](https://www.youtube.com/watch?v=2AnIj_QFtow), and [live video](https://www.youtube.com/watch?v=ISGDrrDcUB0).

### 1. Prerequisites

#### 1.1 [OpenCV](http://opencv.org)
OpenCV 3.1.0 is used in this code.

#### 1.2 [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```shell
sudo apt-get install libeigen3-dev
```

#### 1.3 [Sophus](https://github.com/strasdat/Sophus)
This code use the template implement of Sophus. Recommend the latest released version [v1.0.0](https://github.com/strasdat/Sophus/tree/v1.0.0)

#### 1.4 [glog](https://github.com/google/glog)
This code use glog for logging.
```shell
sudo apt-get install libgoogle-glog-dev
```

#### 1.5 [Ceres](http://ceres-solver.org/installation.html)
Use Ceres-Slover to slove bundle adjustment. Please follow the [installation page](http://ceres-solver.org/installation.html#section-customizing) to install.

#### 1.6 [Pangolin](https://github.com/stevenlovegrove/Pangolin)
This code use Pangolin to display the map reconstructed. When install, just follow the [README.md](https://github.com/stevenlovegrove/Pangolin/blob/master/README.md) file.

#### 1.7 [DBow3](https://github.com/kokerf/DBow3)
After build and install, copy the file `FindDBoW3.cmake` to the directory `cmake_modules`

### 2. Usages & Evalution
the Euroc Dataset is used for evalution. In the project directory, run the demo by the following commond
```shell
./bin/monoVO ./config/euroc.yaml dataset_image_path dataset_csv_file
```
for example, the dataset `MH_01_easy`'s directory is in `/home`, just run
```shell
./bin/monoVO ./config/euroc.yaml ~/MH_01_easy/cam0/data ~/MH_01_easy/cam0/data.csv
```
finally, there will be a `trajectory.txt` saved, and you can use the [evo](https://github.com/MichaelGrupp/evo) to evaluate.