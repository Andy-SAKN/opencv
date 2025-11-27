#pragma once

// 标准库
#include <string>
#include <vector>

// OpenCV 具体模块头文件 (不要用 opencv.hpp)
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp> // 如果你的 hpp 里用到了 resize 等，或者只是为了保险

class YunetWrapper {
public:
    YunetWrapper(const std::string& model_path);
    ~YunetWrapper() = default;

    bool detect(const cv::Mat& img, cv::Rect& out_box);

private:
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float thresh);

private:
    cv::dnn::Net net_;
    std::vector<std::string> out_names_;
    
    const int input_w_ = 640;
    const int input_h_ = 640;
};