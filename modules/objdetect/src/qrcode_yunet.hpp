// qrcode_yunet.hpp

#pragma once
#include <opencv2/core.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <memory>

class YunetWrapper {
public:
    YunetWrapper(const std::string& model_path);

    bool detect(const cv::Mat& img, cv::Rect& out_box);

private:
    cv::Rect scaleBack(const cv::Rect& r, float sx, float sy, int W, int H);
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float thresh);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> opts_;
    std::unique_ptr<Ort::Session> session_;

    int input_w_ = 640;
    int input_h_ = 640;
};

