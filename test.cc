#include "single_image_sr_onnx.h"
#include <chrono>
#include <iostream>

int main() {
  SuperResolution sr;

  // 记录模型加载开始时间
  auto model_load_start = std::chrono::high_resolution_clock::now();
  if (!sr.loadModel("../fsrcnn.onnx")) {
    return 1;
  }
  // 记录模型加载结束时间并计算耗时
  auto model_load_end = std::chrono::high_resolution_clock::now();
  auto model_load_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(model_load_end -
                                                            model_load_start)
          .count();
  std::cout << "模型加载耗时：" << model_load_duration << " 毫秒" << std::endl;

  for (int i = 0; i < 10; i++) {
    auto inference_start = std::chrono::high_resolution_clock::now();
    if (!sr.processImage("../test.jpg", "output.png")) {
      return 1;
    }
    // 记录推理结束时间并计算耗时
    auto inference_end = std::chrono::high_resolution_clock::now();
    auto inference_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(inference_end -
                                                              inference_start)
            .count();
    std::cout << "图像推理耗时：" << inference_duration << " 毫秒" << std::endl;
  }

  return 0;
}