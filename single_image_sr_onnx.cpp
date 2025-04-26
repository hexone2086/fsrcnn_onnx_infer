#include "single_image_sr_onnx.h"
#include <iostream>  // 添加缺失的头文件

SuperResolution::SuperResolution() : env_(ORT_LOGGING_LEVEL_WARNING, "SuperResolution") {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool SuperResolution::loadModel(const std::string& model_path) {
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "模型加载失败：" << e.what() << std::endl;
        return false;
    }
}

bool SuperResolution::processImage(const std::string& input_path, const std::string& output_path) {
    cv::Mat input_image = cv::imread(input_path);
    if (input_image.empty()) {
        std::cerr << "无法读取图像：" << input_path << std::endl;
        return false;
    }

    cv::Mat ycrcb_image;
    cv::cvtColor(input_image, ycrcb_image, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb_image, channels);
    cv::Mat y_channel = channels[0];

    y_channel.convertTo(y_channel, CV_32F, 1.0/255.0);

    std::vector<int64_t> input_shape = {1, 1, y_channel.rows, y_channel.cols};
    std::vector<Ort::Float16_t> input_values(y_channel.total());
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = Ort::Float16_t(y_channel.at<float>(i));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

    // 修复输入输出名称获取方式
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name_ptr.get()};
    const char* output_names[] = {output_name_ptr.get()};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    Ort::Float16_t* output_data = output_tensors[0].GetTensorMutableData<Ort::Float16_t>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    cv::Mat sr_y(output_shape[2], output_shape[3], CV_32F);
    for (size_t i = 0; i < sr_y.total(); ++i) {
        sr_y.at<float>(i) = output_data[i].ToFloat();
    }

    cv::normalize(sr_y, sr_y, 0, 255, cv::NORM_MINMAX);
    sr_y.convertTo(sr_y, CV_8U);

    cv::Mat sr_cr, sr_cb;
    cv::resize(channels[1], sr_cr, sr_y.size());
    cv::resize(channels[2], sr_cb, sr_y.size());

    std::vector<cv::Mat> sr_channels = {sr_y, sr_cr, sr_cb};
    cv::Mat sr_ycrcb, sr_bgr;
    cv::merge(sr_channels, sr_ycrcb);
    cv::cvtColor(sr_ycrcb, sr_bgr, cv::COLOR_YCrCb2BGR);

    return cv::imwrite(output_path, sr_bgr);
}