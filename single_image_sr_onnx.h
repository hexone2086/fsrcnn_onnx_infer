#ifndef SINGLE_IMAGE_SR_ONNX_H
#define SINGLE_IMAGE_SR_ONNX_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>

class SuperResolution {
public:
    SuperResolution();
    bool loadModel(const std::string& model_path);
    bool processImage(const std::string& input_path, const std::string& output_path);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
};

#endif // SINGLE_IMAGE_SR_ONNX_H
