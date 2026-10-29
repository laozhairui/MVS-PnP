#include "yolo_vino.hpp"
#include <chrono>
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;



int main(int argc, char const *argv[])
{   
    // --------- 推理+读取图片 ----------
    //auto logger = std::make_unique<YoloVino::YoloVinoLogger>("../src/YoloVino/config/yolov5fourpoint_vino_config copy.yaml");
    auto logger = std::make_unique<YoloVino::YoloVinoLogger>();
    logger->print_yaml_info();
    logger->set_info_level(YoloVino::LoggerInfoLevel::debug_info);
    YoloVino::Yolov8poseVino vino(std::move(logger));
    //YoloVino::Yolov5fourpointVino vino(std::move(logger));
    cv::Mat ori_img = cv::imread("../src/YoloVino/test_img/infantry_792.png");
    
    // --------- 推理+测速 ----------
    auto t_start = Clock::now();
    std::vector<YoloVino::NNDetectData> results =vino.safe_predict(ori_img, cv::Rect(0, 0, ori_img.cols, ori_img.rows));
    auto t_end = Clock::now();
    std::cout << "[Time] safe_predict total: "
              << std::chrono::duration_cast<us>(t_end - t_start).count()
              << " us" << std::endl;

    // ---------- 可视化 ----------
    for (const auto &det : results)
    {
        // 绘制检测框（绿色）
        cv::rectangle(ori_img, det.rect, cv::Scalar(0, 255, 0), 2);

        // 类别 + 置信度文本
        std::string label = "ID:" + std::to_string(det.class_id) +
                            " conf:" + cv::format("%.2f", det.confidence);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(ori_img,
                      cv::Point(det.rect.x, det.rect.y - labelSize.height - baseLine),
                      cv::Point(det.rect.x + labelSize.width, det.rect.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(ori_img, label,
                    cv::Point(det.rect.x, det.rect.y - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    cv::namedWindow("Detections", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detections",1200,900);
    cv::imshow("Detections", ori_img);
    while (1)
    {
        int key = cv::waitKey(0);
        if (key == 27 || key == 'q')
        {
            break;
        }
        
    }
    cv::destroyAllWindows();
    return 0;
}
