#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    // 加载 ONNX 模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const char* onnx_path = "../fsrcnn.onnx";
    Ort::Session session(env, onnx_path, session_options);

    // 读取低分辨率图像
    std::string lr_image_path = "../test.jpg";
    cv::Mat lr_image = cv::imread(lr_image_path);
    if (lr_image.empty()) {
        std::cerr << "Failed to read image: " << lr_image_path << std::endl;
        return -1;
    }
    std::cout << "Input image size: " << lr_image.cols << "x" << lr_image.rows 
              << " channels: " << lr_image.channels() << std::endl;

    // lr_image.convertTo(lr_image, CV_32F, 1.0 / 255.0);

    // 将 BGR 图像转换为 YCbCr 图像
    cv::Mat lr_ycbcr_image;
    cv::cvtColor(lr_image, lr_ycbcr_image, cv::COLOR_BGR2YCrCb);
    std::cout << "YCbCr image size: " << lr_ycbcr_image.cols << "x" << lr_ycbcr_image.rows 
              << " channels: " << lr_ycbcr_image.channels() << std::endl;

    lr_ycbcr_image.convertTo(lr_ycbcr_image, CV_32F, 1.0 / 255.0);
    // clip to [0, 1]
    cv::minMaxLoc(lr_ycbcr_image, nullptr, nullptr, nullptr, nullptr);

    // 分割 YCbCr 图像数据
    std::vector<cv::Mat> ycrcb_channels;
    cv::split(lr_ycbcr_image, ycrcb_channels);
    cv::Mat lr_y_image = ycrcb_channels[0];
    std::cout << "Y channel size: " << lr_y_image.cols << "x" << lr_y_image.rows 
              << " type: " << lr_y_image.type() << std::endl;

    // FIXME 为什么无法保存
    // cv::imwrite("./input_y_channel.jpg", lr_y_image);
    cv::imshow("Y channel", lr_y_image);


    // 准备输入张量
    std::vector<int64_t> input_shape = {1, 1, lr_y_image.rows, lr_y_image.cols};
    std::cout << "Input tensor shape: [" << input_shape[0] << ", " << input_shape[1] 
              << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;

    std::vector<Ort::Float16_t> input_tensor_values(lr_y_image.total());
    
    // 将 float 转换为 float16
    for (size_t i = 0; i < lr_y_image.total(); ++i) {
        input_tensor_values[i] = Ort::Float16_t(lr_y_image.at<float>(i));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // 获取输入和输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_name_cstr = input_name.get();
    const char* output_name_cstr = output_name.get();

    // 执行推理
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name_cstr, &input_tensor, 1, &output_name_cstr, 1);

    // 处理输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output tensor shape: [" << output_shape[0] << ", " << output_shape[1] 
              << ", " << output_shape[2] << ", " << output_shape[3] << "]" << std::endl;

    cv::Mat sr_y_image(output_shape[2], output_shape[3], CV_32F, output_data);
    std::cout << "SR Y channel size: " << sr_y_image.cols << "x" << sr_y_image.rows 
              << " min: " << *std::min_element(output_data, output_data + sr_y_image.total())
              << " max: " << *std::max_element(output_data, output_data + sr_y_image.total()) << std::endl;

    cv::normalize(sr_y_image, sr_y_image, 0, 1, cv::NORM_MINMAX);

    // 调整 Cr 和 Cb 通道的大小以匹配超分辨率后的 Y 通道
    cv::Mat hr_cb_image, hr_cr_image;
    cv::resize(ycrcb_channels[1], hr_cb_image, cv::Size(sr_y_image.cols, sr_y_image.rows));
    cv::resize(ycrcb_channels[2], hr_cr_image, cv::Size(sr_y_image.cols, sr_y_image.rows));
    std::cout << "Resized Cb/Cr channels size: " << hr_cb_image.cols << "x" << hr_cb_image.rows << std::endl;

    // // 保存 y 通道作为单通道图片
    // std::string sr_y_image_path = "./test_y.jpg";
    // sr_y_image.convertTo(sr_y_image, CV_8U, 255.0);
    // cv::imwrite(sr_y_image_path, sr_y_image);

    // 合并 YCbCr 通道
    std::vector<cv::Mat> sr_ycrcb_channels = {sr_y_image, hr_cr_image, hr_cb_image};
    cv::Mat sr_ycbcr_image;
    cv::merge(sr_ycrcb_channels, sr_ycbcr_image);

    // 将 YCbCr 图像转换为 BGR 图像
    cv::Mat sr_image;
    cv::cvtColor(sr_ycbcr_image, sr_image, cv::COLOR_YCrCb2BGR);

    // 保存超分辨率结果
    std::string sr_image_path = "./test.jpg";
    sr_image.convertTo(sr_image, CV_8U, 255.0);
    cv::imwrite(sr_image_path, sr_image);

    std::cout << "Super-resolution result saved to: " << sr_image_path << std::endl;

    return 0;
}
