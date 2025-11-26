#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    const char* model = std::getenv("OPENCV_YUNET_MODEL");
    if (!model) {
        std::cout << "[ERROR] Please set OPENCV_YUNET_MODEL first\n";
        return -1;
    }
    std::cout << "[INFO] Using YUNET model: " << model << std::endl;

    cv::QRCodeDetector qr;

    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cout << "Failed to read test.jpg\n";
        return -1;
    }

    cv::Mat points, straight;
    std::string data = qr.detectAndDecode(img, points, straight);

    if (data.empty()) {
        std::cout << "[RESULT] Decode failed\n";
    } else {
        std::cout << "[RESULT] QR Content: " << data << std::endl;
    }

    if (!points.empty()) {
        std::cout << "[INFO] YUNET detected points:\n";
        std::cout << points << std::endl;

        // draw result
        std::vector<cv::Point> pts;
        for (int i = 0; i < 4; ++i) {
            pts.emplace_back(points.at<cv::Point2f>(i));
        }
        for (int i = 0; i < 4; i++) {
            cv::line(img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("result.jpg", img);
        std::cout << "[INFO] Saved result.jpg\n";
    }

    return 0;
}
