cmake_minimum_required(VERSION 3.8)
project(fsrcnn)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

# Set OpenCV path (using opencv-mobile)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check OpenCV_DIR or install OpenCV.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 直接设置 ONNX Runtime 头文件包含路径和动态链接库路径
set(ONNXRuntime_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/onnxruntime/include)
set(ONNXRuntime_LIBRARIES ${CMAKE_SOURCE_DIR}/onnxruntime/lib64/libonnxruntime.so)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
)

# Add executable
add_executable(fsrcnn single_image_sr_onnx.cpp)

# Link libraries
target_link_libraries(fsrcnn
    ${OpenCV_LIBS}
    ${ONNXRuntime_LIBRARIES}
)