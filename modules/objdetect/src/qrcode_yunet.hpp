#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>  // OpenCV DNN header

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
    cv::dnn::Net net_;  // OpenCV DNN model

    int input_w_ = 640;
    int input_h_ = 640;
};
