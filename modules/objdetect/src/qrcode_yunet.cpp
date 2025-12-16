#include "qrcode_yunet.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

// 辅助函数：Sigmoid
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

YunetWrapper::YunetWrapper(const std::string& model_path)
{
    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading Yunet model: " << e.what() << std::endl;
        return;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    out_names_ = net_.getUnconnectedOutLayersNames();
}

std::vector<int> YunetWrapper::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float thresh)
{
    std::vector<int> idx;
    if (boxes.empty()) return idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, thresh, idx);
    return idx;
}

bool YunetWrapper::detect(const cv::Mat& img, cv::Rect& out_box)
{
    if (net_.empty() || img.empty()) return false;

    // 1. 预处理 (Letterbox)
    int w = img.cols;
    int h = img.rows;
    float scale = std::min((float)input_w_ / w, (float)input_h_ / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    cv::Mat resized;
    if (w != new_w || h != new_h) cv::resize(img, resized, cv::Size(new_w, new_h));
    else resized = img;

    cv::Mat input_blob_img;
    cv::copyMakeBorder(resized, input_blob_img, 
                       dh, input_h_ - new_h - dh, 
                       dw, input_w_ - new_w - dw, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat blob = cv::dnn::blobFromImage(input_blob_img, 1.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);
    net_.setInput(blob);

    // 2. 推理
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);

    std::map<std::string, cv::Mat> out_map;
    for (size_t i = 0; i < out_names_.size(); ++i) out_map[out_names_[i]] = outs[i];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> strides = {8, 16, 32};

    // 3. 解码 (适配 Flattened Output)
    for (int stride : strides)
    {
        std::string layer_box = "bbox_" + std::to_string(stride);
        std::string layer_cls = "cls_" + std::to_string(stride);
        std::string layer_obj = "obj_" + std::to_string(stride);

        if (out_map.find(layer_box) == out_map.end()) continue;

        const cv::Mat& box_mat = out_map[layer_box];
        const cv::Mat& cls_mat = out_map[layer_cls];
        const cv::Mat& obj_mat = out_map[layer_obj];

        // 计算当前 stride 下原本应该有的 grid 尺寸
        int grid_w = input_w_ / stride; // e.g., 640/8 = 80
        int grid_h = input_h_ / stride; // e.g., 640/8 = 80
        int num_anchors = grid_w * grid_h;

        // 获取数据指针 (假定数据是 float 并且连续)
        const float* ptr_box = (float*)box_mat.data;
        const float* ptr_cls = (float*)cls_mat.data;
        const float* ptr_obj = (float*)obj_mat.data;

        // 自动检测维度布局
        // 如果是 [1, C, H, W]，step 通常很大
        // 如果是 [1, N, C]，step 通常是 C
        // 这里我们采用最稳健的方法：按照总元素数量 num_anchors 遍历
        // 并根据 total_size / num_anchors 算出每个 anchor 占用的步长
        
        int total_elements_box = box_mat.total();
        int step_box = total_elements_box / num_anchors; // 应该是 4

        int total_elements_cls = cls_mat.total();
        int step_cls = total_elements_cls / num_anchors; // 应该是 classes数量

        int total_elements_obj = obj_mat.total();
        int step_obj = total_elements_obj / num_anchors; // 应该是 1

        for (int i = 0; i < num_anchors; i++)
        {
            // 1. 计算当前 anchor 在 grid 中的逻辑坐标 (关键修正!)
            int grid_y = i / grid_w;
            int grid_x = i % grid_w;

            // 2. 获取 Objectness
            // 注意：如果 step_obj=1，则直接 ptr_obj[i]。如果是 NCHW 格式，这里逻辑可能不同，
            // 但鉴于你的 log 报错，基本确定是 Flattened 格式 (N,1) 或 (1,N,1)
            float obj_score = sigmoid(ptr_obj[i * step_obj]);
            if (obj_score < 0.3f) continue;

            // 3. 获取 Class Score
            // 假设我们只关心最大值
            float max_cls_score = 0.f;
            for (int c = 0; c < step_cls; c++) {
                float s = sigmoid(ptr_cls[i * step_cls + c]);
                if (s > max_cls_score) max_cls_score = s;
            }

            float final_score = max_cls_score * obj_score;
            if (final_score < 0.3f) continue;

            // 4. Decode Box
            // 指针偏移：第 i 个 anchor，加上 0~3 的 offset
            float r0 = ptr_box[i * step_box + 0]; // x
            float r1 = ptr_box[i * step_box + 1]; // y
            float r2 = ptr_box[i * step_box + 2]; // w
            float r3 = ptr_box[i * step_box + 3]; // h

            // 关键修正：这里必须使用 grid_x/grid_y，而不是循环变量 i
            float cx = (r0 * stride) + (grid_x * stride);
            float cy = (r1 * stride) + (grid_y * stride);

            float w_pred = std::exp(r2) * stride;
            float h_pred = std::exp(r3) * stride;

            float x1 = cx - w_pred / 2.0f;
            float y1 = cy - h_pred / 2.0f;

            if (w_pred > 0 && h_pred > 0) {
                boxes.emplace_back(x1, y1, w_pred, h_pred);
                scores.push_back(final_score);
            }
        }
    }

    if (boxes.empty()) return false;

    // 4. NMS
    std::vector<int> keep = nms(boxes, scores, 0.45f);
    if (keep.empty()) return false;

    cv::Rect best_box_net = boxes[keep[0]];

    // 5. 坐标还原
    float x_final = (best_box_net.x - dw) / scale;
    float y_final = (best_box_net.y - dh) / scale;
    float w_final = best_box_net.width / scale;
    float h_final = best_box_net.height / scale;

    int x1 = std::max(0, (int)x_final);
    int y1 = std::max(0, (int)y_final);
    int x2 = std::min(w, (int)(x_final + w_final));
    int y2 = std::min(h, (int)(y_final + h_final));

    out_box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    
    // 增加一点鲁棒性
    return (out_box.width > 2 && out_box.height > 2);
}


// ============================================================
// detectMulti
// ------------------------------------------------------------
// 与 detect 完全一致的前处理 / decode / NMS
// 唯一差异：
//   1. NMS 后保留所有 keep 的框
//   2. 每个框都做 inverse letterbox
// ============================================================
bool YunetWrapper::detectMulti(
    const cv::Mat& img,
    std::vector<cv::Rect>& out_boxes)
{
    out_boxes.clear();
    if (net_.empty() || img.empty())
        return false;

    // ----------------------------
    // 1. Letterbox 预处理（与 detect 完全一致）
    // ----------------------------
    int w = img.cols;
    int h = img.rows;

    float scale = std::min((float)input_w_ / w, (float)input_h_ / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);

    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    cv::Mat resized;
    if (w != new_w || h != new_h)
        cv::resize(img, resized, cv::Size(new_w, new_h));
    else
        resized = img;

    cv::Mat input_blob_img;
    cv::copyMakeBorder(
        resized,
        input_blob_img,
        dh, input_h_ - new_h - dh,
        dw, input_w_ - new_w - dw,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    cv::Mat blob = cv::dnn::blobFromImage(
        input_blob_img,
        1.0,
        cv::Size(),
        cv::Scalar(0, 0, 0),
        false,
        false
    );

    net_.setInput(blob);

    // ----------------------------
    // 2. 推理
    // ----------------------------
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);

    std::map<std::string, cv::Mat> out_map;
    for (size_t i = 0; i < out_names_.size(); ++i)
        out_map[out_names_[i]] = outs[i];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    std::vector<int> strides = {8, 16, 32};

    // ----------------------------
    // 3. Decode（与 detect 完全相同）
    // ----------------------------
    for (int stride : strides)
    {
        std::string layer_box = "bbox_" + std::to_string(stride);
        std::string layer_cls = "cls_"  + std::to_string(stride);
        std::string layer_obj = "obj_"  + std::to_string(stride);

        if (out_map.find(layer_box) == out_map.end())
            continue;

        const cv::Mat& box_mat = out_map[layer_box];
        const cv::Mat& cls_mat = out_map[layer_cls];
        const cv::Mat& obj_mat = out_map[layer_obj];

        int grid_w = input_w_ / stride;
        int grid_h = input_h_ / stride;
        int num_anchors = grid_w * grid_h;

        const float* ptr_box = (float*)box_mat.data;
        const float* ptr_cls = (float*)cls_mat.data;
        const float* ptr_obj = (float*)obj_mat.data;

        int step_box = box_mat.total() / num_anchors;
        int step_cls = cls_mat.total() / num_anchors;
        int step_obj = obj_mat.total() / num_anchors;

        for (int i = 0; i < num_anchors; i++)
        {
            int grid_y = i / grid_w;
            int grid_x = i % grid_w;

            float obj_score = sigmoid(ptr_obj[i * step_obj]);
            if (obj_score < 0.3f) continue;

            float max_cls_score = 0.f;
            for (int c = 0; c < step_cls; c++) {
                float s = sigmoid(ptr_cls[i * step_cls + c]);
                if (s > max_cls_score)
                    max_cls_score = s;
            }

            float final_score = max_cls_score * obj_score;
            if (final_score < 0.3f) continue;

            float r0 = ptr_box[i * step_box + 0];
            float r1 = ptr_box[i * step_box + 1];
            float r2 = ptr_box[i * step_box + 2];
            float r3 = ptr_box[i * step_box + 3];

            float cx = r0 * stride + grid_x * stride;
            float cy = r1 * stride + grid_y * stride;

            float bw = std::exp(r2) * stride;
            float bh = std::exp(r3) * stride;

            boxes.emplace_back(
                cx - bw * 0.5f,
                cy - bh * 0.5f,
                bw,
                bh
            );
            scores.push_back(final_score);
        }
    }

    if (boxes.empty())
        return false;

    // ----------------------------
    // 4. NMS（保留全部 keep）
    // ----------------------------
    std::vector<int> keep = nms(boxes, scores, 0.45f);
    if (keep.empty())
        return false;

    const int MAX_BOXES = 10;

    // 按 score 从高到低排序 keep
    std::sort(keep.begin(), keep.end(),
            [&](int a, int b) {
                return scores[a] > scores[b];
            });

    // 如果超过 10 个，只保留前 10 个
    if ((int)keep.size() > MAX_BOXES)
        keep.resize(MAX_BOXES);

    // ----------------------------
    // 5. 所有框做 inverse letterbox
    // ----------------------------
    for (int idx : keep)
    {
        const cv::Rect& b = boxes[idx];

        float x = (b.x - dw) / scale;
        float y = (b.y - dh) / scale;
        float w0 = b.width  / scale;
        float h0 = b.height / scale;

        int x1 = std::max(0, (int)x);
        int y1 = std::max(0, (int)y);
        int x2 = std::min(w, (int)(x + w0));
        int y2 = std::min(h, (int)(y + h0));

        if (x2 > x1 && y2 > y1)
            out_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }

    return !out_boxes.empty();
}
