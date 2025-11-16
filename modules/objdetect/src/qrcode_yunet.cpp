#include "qrcode_yunet.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>


// ===================================================
// Constructor
// ===================================================
YunetWrapper::YunetWrapper(const std::string& model_path)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yunet");
    opts_ = std::make_unique<Ort::SessionOptions>();

    opts_->SetIntraOpNumThreads(1);
    opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *opts_);
}


// ===================================================
// Helper: scale box back to original resolution
// ===================================================
cv::Rect YunetWrapper::scaleBack(const cv::Rect& r,
                                 float sx, float sy,
                                 int W, int H)
{
    int x1 = std::max(0, (int)(r.x * sx));
    int y1 = std::max(0, (int)(r.y * sy));
    int x2 = std::min(W, (int)((r.x + r.width) * sx));
    int y2 = std::min(H, (int)((r.y + r.height) * sy));
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}


// ===================================================
// NMS wrapper
// ===================================================
std::vector<int> YunetWrapper::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float thresh)
{
    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, thresh, idx);
    return idx;
}


bool YunetWrapper::detect(const cv::Mat& img, cv::Rect& out_box)
{
    if (!session_) return false;

    int W = img.cols;
    int H = img.rows;

    // ----------------------------------------
    // Preprocess: resize → float32 → CHW
    // ----------------------------------------
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w_, input_h_));

    resized.convertTo(resized, CV_32F);

    // HWC → CHW
    std::vector<float> blob;
    blob.resize(3 * input_w_ * input_h_);

    int idx = 0;
    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < input_h_; y++)
        {
            for (int x = 0; x < input_w_; x++)
            {
                blob[idx++] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    // Create ONNX tensor
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> dims = {1, 3, input_h_, input_w_};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        blob.data(),
        blob.size(),
        dims.data(),
        dims.size()
    );

    // ----------------------------------------
    // Prepare input/output names
    // ----------------------------------------
    Ort::AllocatorWithDefaultOptions allocator;

    // Input name
    Ort::AllocatedStringPtr inName =
        session_->GetInputNameAllocated(0, allocator);

    const char* input_names[] = { inName.get() };   // ✓ 正确写法

    // Output names
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> outNameHolders;

    size_t outCount = session_->GetOutputCount();
    for (size_t i = 0; i < outCount; i++)
    {
        outNameHolders.push_back(session_->GetOutputNameAllocated(i, allocator));
        outputNames.push_back(outNameHolders.back().get());
    }

    // ----------------------------------------
    // Run ONNX model (✓ 已修复)
    // ----------------------------------------
    auto out = session_->Run(
        Ort::RunOptions{},
        input_names,
        &inputTensor,
        1,
        outputNames.data(),
        outputNames.size()
    );

    // ----------------------------------------
    // Decode YUNET output
    // ----------------------------------------
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    int strides[3] = {8, 16, 32};
    int outIndex = 0;

    for (int s = 0; s < 3; s++)
    {
        int stride = strides[s];

        auto& cls   = out[outIndex++];
        auto& obj   = out[outIndex++];
        auto& box   = out[outIndex++];
        auto& kps   = out[outIndex++];

        float* cls_ptr = cls.GetTensorMutableData<float>();
        float* obj_ptr = obj.GetTensorMutableData<float>();
        float* box_ptr = box.GetTensorMutableData<float>();

        auto ts = cls.GetTensorTypeAndShapeInfo();
        int N = ts.GetElementCount() / 5;

        int gw = input_w_ / stride;
        int gh = input_h_ / stride;

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
