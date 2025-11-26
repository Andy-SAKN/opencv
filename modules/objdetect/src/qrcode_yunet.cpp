#include <iostream>  // 添加了这个头文件

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

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
    cv::dnn::Net net_;  // 使用 OpenCV DNN 模型

    int input_w_ = 640;
    int input_h_ = 640;
};

// 构造函数
YunetWrapper::YunetWrapper(const std::string& model_path)
{
    // 使用OpenCV加载DNN模型
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (net_.empty()) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
    }
}

// 辅助函数：将框坐标转换回原始分辨率
cv::Rect YunetWrapper::scaleBack(const cv::Rect& r, float sx, float sy, int W, int H)
{
    int x1 = std::max(0, (int)(r.x * sx));
    int y1 = std::max(0, (int)(r.y * sy));
    int x2 = std::min(W, (int)((r.x + r.width) * sx));
    int y2 = std::min(H, (int)((r.y + r.height) * sy));
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// 辅助函数：非极大值抑制
std::vector<int> YunetWrapper::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float thresh)
{
    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, thresh, idx);
    return idx;
}

// 检测函数
bool YunetWrapper::detect(const cv::Mat& img, cv::Rect& out_box)
{
    if (net_.empty()) return false;

    int W = img.cols;
    int H = img.rows;

    // ----------------------------------------
    // 预处理：调整大小 → 转换为 float32 → CHW
    // ----------------------------------------
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w_, input_h_));

    resized.convertTo(resized, CV_32F);

    // HWC → CHW
    cv::Mat blob = cv::dnn::blobFromImage(resized);

    // 将输入设置到网络中
    net_.setInput(blob);

    // ----------------------------------------
    // 执行 OpenCV DNN 推理
    // ----------------------------------------
    cv::Mat output = net_.forward();

    // ----------------------------------------
    // 解码 YUNET 输出
    // ----------------------------------------
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    int strides[3] = {8, 16, 32};
    int outIndex = 0;

    for (int stride_idx = 0; stride_idx < 3; stride_idx++)  // 重命名变量s
    {
        int stride = strides[stride_idx];

        auto cls = output.row(outIndex++);
        auto obj = output.row(outIndex++);
        auto box = output.row(outIndex++);
        auto kps = output.row(outIndex++);  // 如果不使用，可以删除

        float* cls_ptr = cls.ptr<float>();
        float* obj_ptr = obj.ptr<float>();
        float* box_ptr = box.ptr<float>();

        int N = cls.total() / 5;

        int gw = input_w_ / stride;
        // gh 已经不再使用

        for (int i = 0; i < N; i++)
        {
            if (obj_ptr[i] < 0.2f) continue;

            float maxc = -1.f;
            int cid = -1;
            for (int c = 0; c < 5; c++)
            {
                float s = cls_ptr[i * 5 + c];
                if (s > maxc)
                {
                    maxc = s;
                    cid = c;
                }
            }

            float score = maxc * obj_ptr[i];
            if (score < 0.2f) continue;
            if (cid != 3) continue;

            int y = i / gw;
            int x = i % gw;

            float ax = x * stride;
            float ay = y * stride;

            float dx = box_ptr[i * 4 + 0];
            float dy = box_ptr[i * 4 + 1];
            float dw = box_ptr[i * 4 + 2];
            float dh = box_ptr[i * 4 + 3];

            float cx = dx * stride + ax;
            float cy = dy * stride + ay;

            float ww = std::exp(dw) * stride;
            float hh = std::exp(dh) * stride;

            boxes.emplace_back(cx - ww*0.5f, cy - hh*0.5f, ww, hh);
            scores.push_back(score);
        }
    }

    if (boxes.empty()) return false;

    auto keep = nms(boxes, scores, 0.45f);
    if (keep.empty()) return false;

    cv::Rect best = boxes[keep[0]];

    float sx = (float)W / input_w_;
    float sy = (float)H / input_h_;

    out_box = scaleBack(best, sx, sy, W, H);
    return true;
}
