#include "qrcode_yunet.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

// ===================================================
// Constructor: Load ONNX using OpenCV DNN
// ===================================================
YunetWrapper::YunetWrapper(const std::string& model_path)
{
    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading Yunet model: " << e.what() << std::endl;
        return;
    }

    // 设置后端和目标设备 (根据需要可以改为 CUDA)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 获取输出层名称，避免每次推理都获取
    out_names_ = net_.getUnconnectedOutLayersNames();
}

// ===================================================
// NMS Wrapper
// ===================================================
std::vector<int> YunetWrapper::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float thresh)
{
    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, thresh, idx);
    return idx;
}

// ===================================================
// Detect Function
// ===================================================
bool YunetWrapper::detect(const cv::Mat& img, cv::Rect& out_box)
{
    if (net_.empty() || img.empty()) return false;

    // ----------------------------------------
    // 1. Letterbox Preprocess (保持比例补黑边)
    // ----------------------------------------
    int w = img.cols;
    int h = img.rows;

    float scale = std::min((float)input_w_ / w, (float)input_h_ / h);
    
    // 计算缩放后的新尺寸
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);

    // 计算 padding (让图像居中)
    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    // Resize 图像
    cv::Mat resized;
    if (w != new_w || h != new_h) {
        cv::resize(img, resized, cv::Size(new_w, new_h));
    } else {
        resized = img;
    }

    // 填充黑边 (Top, Bottom, Left, Right)
    cv::Mat input_blob_img;
    cv::copyMakeBorder(resized, input_blob_img, 
                       dh, input_h_ - new_h - dh, 
                       dw, input_w_ - new_w - dw, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // ----------------------------------------
    // 2. Convert to Blob (HWC -> CHW, BGR -> RGB)
    // ----------------------------------------
    // scalefactor=1.0: 因为之前的代码是 convertTo(CV_32F) 而没有除以 255，说明模型输入是 0-255
    // swapRB=true:     用户要求输入 RGB，而 OpenCV 读取是 BGR
    // crop=false:      我们已经手动处理了尺寸
    cv::Mat blob = cv::dnn::blobFromImage(input_blob_img, 1.0, cv::Size(), cv::Scalar(0, 0, 0), true, false);

    net_.setInput(blob);

    // ----------------------------------------
    // 3. Run Inference
    // ----------------------------------------
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);

    // ----------------------------------------
    // 4. Decode Output
    // ----------------------------------------
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    int strides[3] = {8, 16, 32};
    int outIndex = 0;

    // 假设输出顺序与之前一致：每层 stride 有 4 个 tensor (cls, obj, box, kps)
    // 注意：如果 cv::dnn 解析 ONNX 的层顺序与 ONNXRuntime 不一致，这里可能需要根据 layer name 调整
    // 但通常 getUnconnectedOutLayersNames 的顺序是确定的。
    
    // 安全检查
    if (outs.size() < 12) return false; 

    for (int s = 0; s < 3; s++)
    {
        int stride = strides[s];

        // 获取当前 stride 的 4 个输出
        // 注意：cv::dnn 的输出 Mat 形状通常是 [batch, channels, h, w] 或平铺的
        // 这里假设与 ONNXRuntime 的解析逻辑一致 (NCHW 或类似的扁平结构)
        const float* cls_ptr = outs[outIndex++].ptr<float>();
        const float* obj_ptr = outs[outIndex++].ptr<float>();
        const float* box_ptr = outs[outIndex++].ptr<float>();
        outIndex++; // 跳过 kps，因为不需要

        // 计算特征图大小
        int gw = input_w_ / stride;
        int gh = input_h_ / stride;
        int N = gw * gh;

        for (int i = 0; i < N; i++)
        {
            float obj_score = obj_ptr[i];
            if (obj_score < 0.2f) continue;

            float maxc = -1.f;
            int cid = -1;
            // 假设有 5 个类别 (根据原代码逻辑)
            for (int c = 0; c < 5; c++)
            {
                float s = cls_ptr[i * 5 + c];
                if (s > maxc)
                {
                    maxc = s;
                    cid = c;
                }
            }

            // 假设 class 3 是二维码
            if (cid != 3) continue;

            float score = maxc * obj_score;
            if (score < 0.2f) continue;

            // 还原网格坐标
            int y = i / gw;
            int x = i % gw;

            // 回归 box
            float ax = x * stride;
            float ay = y * stride;

            float dx_val = box_ptr[i * 4 + 0];
            float dy_val = box_ptr[i * 4 + 1];
            float dw_val = box_ptr[i * 4 + 2];
            float dh_val = box_ptr[i * 4 + 3];

            float cx = dx_val * stride + ax;
            float cy = dy_val * stride + ay;

            float ww = std::exp(dw_val) * stride;
            float hh = std::exp(dh_val) * stride;

            boxes.emplace_back(cx - ww * 0.5f, cy - hh * 0.5f, ww, hh);
            scores.push_back(score);
        }
    }

    if (boxes.empty()) return false;

    // ----------------------------------------
    // 5. NMS
    // ----------------------------------------
    auto keep = nms(boxes, scores, 0.45f);
    if (keep.empty()) return false;

    cv::Rect best_box_net = boxes[keep[0]];

    // ----------------------------------------
    // 6. Scale Back to Original Image (Inverse Letterbox)
    // ----------------------------------------
    // 坐标映射公式: x_orig = (x_net - padding) / scale
    
    float x = (best_box_net.x - dw) / scale;
    float y = (best_box_net.y - dh) / scale;
    float w_orig = best_box_net.width / scale;
    float h_orig = best_box_net.height / scale;

    // 边界保护，防止越界
    int x1 = std::max(0, (int)x);
    int y1 = std::max(0, (int)y);
    int x2 = std::min(w, (int)(x + w_orig));
    int y2 = std::min(h, (int)(y + h_orig));

    out_box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    
    // 简单的有效性检查
    if (out_box.width <= 0 || out_box.height <= 0) return false;

    return true;
}