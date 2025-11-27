#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class YunetWrapper {
public:
    YunetWrapper(const std::string& model_path);
    ~YunetWrapper() = default;

    // 返回 true 表示检测到了目标
    bool detect(const cv::Mat& img, cv::Rect& out_box);

private:
    cv::dnn::Net net_;
    std::vector<std::string> out_names_;
    
    // 模型输入尺寸 (用户指定 640)
    const int input_w_ = 640;
    const int input_h_ = 640;

    // NMS 辅助
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float thresh);
};