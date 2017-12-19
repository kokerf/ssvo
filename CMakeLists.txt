cmake_minimum_required(VERSION 2.8.3)
project(ssvo)

## -----------------------
## User's option
## -----------------------
set(DEBUG_OUTPUT TRUE)
message(STATUS "Imformation output: " ${DEBUG_OUTPUT})

# Definitions
if(DEBUG_OUTPUT)
  add_definitions(-DSSVO_IMFORM_OUTPUT)
endif()

## -----------------------
## Build setting
## -----------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -mmmx -msse -msse -msse2 -msse3 -mssse3")

## -----------------------
## Library required
## -----------------------
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV 3.1.0 REQUIRED)
if(OpenCV_FOUND)
    message("-- Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "-- Can Not Found OpenCV3")
endif()

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
FIND_PACKAGE(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# glog
find_package(glog 0.3.5 REQUIRED)
#include_directories(${GLOG_INCLUDE_DIR})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Pangolin
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/fast/include/
)

LIST(APPEND LINK_LIBS
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${GLOG_LIBRARY}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/fast/build/libfast.so
)

## -----------------------
## Build library
## -----------------------

# Set sourcefiles
LIST(APPEND SOURCEFILES
    src/camera.cpp
    src/map_point.cpp
    src/frame.cpp
    src/keyframe.cpp
    src/map.cpp
    src/feature_detector.cpp
    src/feature_tracker.cpp
    src/initializer.cpp
    src/optimizer.cpp
    src/alignment.cpp
    src/local_mapping.cpp
    src/system.cpp
    src/viewer.cpp)

#add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})
#target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

## -----------------------
## Build test
## -----------------------
add_executable(test_feature_detector test/test_feature_detector.cpp ${SOURCEFILES})
target_link_libraries(test_feature_detector ${LINK_LIBS})

add_executable(test_initializer_seq test/test_initializer_seq.cpp src/initializer.cpp ${SOURCEFILES})
target_link_libraries(test_initializer_seq  ${LINK_LIBS})

add_executable(test_glog test/test_glog.cpp)
target_link_libraries(test_glog  ${LINK_LIBS})

add_executable(test_utils test/test_utils.cpp)
target_link_libraries(test_utils  ${LINK_LIBS})

add_executable(test_alignment test/test_alignment.cpp ${SOURCEFILES})
target_link_libraries(test_alignment  ${LINK_LIBS})

add_executable(test_alignment_2d test/test_alignment_2d.cpp ${SOURCEFILES})
target_link_libraries(test_alignment_2d  ${LINK_LIBS})

## -----------------------
## Build VO
## -----------------------
add_executable(monoVO demo/monoVO.cpp ${SOURCEFILES})
target_link_libraries(monoVO ${LINK_LIBS})