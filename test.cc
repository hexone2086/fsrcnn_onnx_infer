#include "single_image_sr_onnx.h"

int main() {
    SuperResolution sr;
    if (!sr.loadModel("../fsrcnn.onnx")) {
        return 1;
    }
    if (!sr.processImage("../test.jpg", "output.png")) {
        return 1;
    }
    return 0;
}