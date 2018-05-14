cmake_minimum_required(VERSION 2.8.3)
project(ssvo)

## -----------------------
## User's option
## -----------------------
option(SSVO_TEST_ENABLE "If build the test files." TRUE)
option(SSVO_DBOW_ENABLE "If use the DBoW library." OFF)
option(SSVO_TRACE_ENABLE "If use the time tracing." TRUE)
message(STATUS "Test Enable  : "   ${SSVO_TEST_ENABLE})
message(STATUS "DBoW Enable  : "   ${SSVO_DBOW_ENABLE})
message(STATUS "Trace Enable : "   ${SSVO_TRACE_ENABLE})

# Definitions
if(SSVO_TRACE_ENABLE)
    add_definitions(-DSSVO_USE_TRACE)
endif()

if(SSVO_DBOW_ENABLE)
    add_definitions(-DSSVO_DBOW_ENABLE)
endif()

## -----------------------
## Build setting
## -----------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build Type  : " ${CMAKE_BUILD_TYPE})

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

IF(MSVC)
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  ADD_DEFINITIONS(-D__SSE2__)
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -mmmx -msse -msse -msse2 -msse3 -mssse3")

## -----------------------
## Library required
## -----------------------
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# fast
LIST(APPEND CMAKE_MODULE_PATH  ${PROJECT_SOURCE_DIR}/Thirdparty/fast/build)
find_package(fast REQUIRED)
include_directories(${fast_INCLUDE_DIR})

# OpenCV
find_package(OpenCV 3.1.0 REQUIRED)
if(OpenCV_FOUND)
    message("-- Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "-- Can Not Found OpenCV3")
endif()

# Eigen
find_package(Eigen 3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
FIND_PACKAGE(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# glog
find_package(Glog 0.3.5 REQUIRED)
#include_directories(${GLOG_INCLUDE_DIR})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# DBoW3
if(SSVO_DBOW_ENABLE)
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

LIST(APPEND LINK_LIBS
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${GLOG_LIBRARY}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${DBoW3_LIBRARIES}
    ${fast_LIBRARY}
)

## -----------------------
## Build library
## -----------------------

# Set sourcefiles
LIST(APPEND SOURCEFILES
    src/camera.cpp
    src/map_point.cpp
    src/seed.cpp
    src/frame.cpp
    src/keyframe.cpp
    src/map.cpp
    src/utils.cpp
    src/feature_detector.cpp
    src/feature_tracker.cpp
    src/feature_alignment.cpp
    src/image_alignment.cpp
    src/initializer.cpp
    src/optimizer.cpp
    src/depth_filter.cpp
    src/local_mapping.cpp
    src/system.cpp
    src/viewer.cpp
    src/brief.cpp
)

add_library(${PROJECT_NAME} STATIC ${SOURCEFILES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

## -----------------------
## Build test
## -----------------------
if(SSVO_TEST_ENABLE)
add_executable(test_feature_detector test/test_feature_detector.cpp)
target_link_libraries(test_feature_detector ${PROJECT_NAME})

add_executable(test_initializer_seq test/test_initializer_seq.cpp src/initializer.cpp)
target_link_libraries(test_initializer_seq ${PROJECT_NAME})

add_executable(test_glog test/test_glog.cpp)
target_link_libraries(test_glog  ${PROJECT_NAME})

add_executable(test_utils test/test_utils.cpp)
target_link_libraries(test_utils ${PROJECT_NAME})

add_executable(test_alignment test/test_alignment.cpp)
target_link_libraries(test_alignment ${PROJECT_NAME})

add_executable(test_alignment_2d test/test_alignment_2d.cpp src/feature_alignment.cpp)
target_link_libraries(test_alignment_2d ${PROJECT_NAME})

add_executable(test_triangulation test/test_triangulation.cpp)
target_link_libraries(test_triangulation ${PROJECT_NAME})

add_executable(test_pattern test/test_parttern.cpp)
target_link_libraries(test_pattern ${PROJECT_NAME})

add_executable(test_optimizer test/test_optimizer.cpp)
target_link_libraries(test_optimizer ${PROJECT_NAME})

add_executable(test_camera_model test/test_camera_model.cpp src/camera.cpp)
target_link_libraries(test_camera_model ${LINK_LIBS})

add_executable(test_timer test/test_timer.cpp)

if(SSVO_DBOW_ENABLE)
add_executable(test_dbow3 test/test_dbow3.cpp)
target_link_libraries(test_dbow3 ${PROJECT_NAME})
endif()
endif(SSVO_TEST_ENABLE)

## -----------------------
## Build VO
## -----------------------
add_executable(monoVO_dataset demo/monoVO_dataset.cpp)
target_link_libraries(monoVO_dataset ${PROJECT_NAME})

add_executable(monoVO_live demo/monoVO_live.cpp)
target_link_libraries(monoVO_live ${PROJECT_NAME})