#include "Camera.h"
#include "yolo_vino.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct PnPResult
{
    double yaw = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double distance = 0.0;
    double depth = 0.0;
    cv::Vec3d translation{0.0, 0.0, 0.0};
};

static bool solve_armor_pose(
    const std::vector<cv::Point3f>& keypoints,
    double armor_width,
    double armor_height,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    PnPResult& result)
{
    if (keypoints.size() != 4 ||
        camera_matrix.empty() ||
        dist_coeffs.empty() ||
        armor_width <= 0.0 ||
        armor_height <= 0.0)
    {
        return false;
    }

    const float half_width = static_cast<float>(armor_width / 2.0);
    const float half_height = static_cast<float>(armor_height / 2.0);

    const std::vector<cv::Point3f> object_points = {
        {-half_width, -half_height, 0.0F},
        { half_width, -half_height, 0.0F},
        { half_width,  half_height, 0.0F},
        {-half_width,  half_height, 0.0F}
    };

    std::vector<cv::Point2f> image_points;
    image_points.reserve(4);

    for (const auto& point : keypoints)
    {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y))
        {
            return false;
        }

        image_points.emplace_back(point.x, point.y);
    }

    cv::Vec3d rotation_vector;
    cv::Vec3d translation_vector;

    bool solved = false;

    try
    {
        solved = cv::solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rotation_vector,
            translation_vector,
            false,
            cv::SOLVEPNP_ITERATIVE
        );
    }
    catch (const cv::Exception&)
    {
        return false;
    }

    if (!solved ||
        !std::isfinite(translation_vector[0]) ||
        !std::isfinite(translation_vector[1]) ||
        !std::isfinite(translation_vector[2]) ||
        translation_vector[2] <= 0.0)
    {
        return false;
    }

    cv::Mat rotation_matrix;

    try
    {
        cv::Rodrigues(rotation_vector, rotation_matrix);
    }
    catch (const cv::Exception&)
    {
        return false;
    }

    if (rotation_matrix.rows != 3 ||
        rotation_matrix.cols != 3 ||
        rotation_matrix.type() != CV_64F)
    {
        return false;
    }

    const cv::Matx33d rotation_world_to_camera(
        rotation_matrix.at<double>(0, 0),
        rotation_matrix.at<double>(0, 1),
        rotation_matrix.at<double>(0, 2),
        rotation_matrix.at<double>(1, 0),
        rotation_matrix.at<double>(1, 1),
        rotation_matrix.at<double>(1, 2),
        rotation_matrix.at<double>(2, 0),
        rotation_matrix.at<double>(2, 1),
        rotation_matrix.at<double>(2, 2)
    );

    const cv::Matx33d rotation_camera_to_world =
        rotation_world_to_camera.t();

    const double q11 = rotation_camera_to_world(0, 0);
    const double q12 = rotation_camera_to_world(0, 1);
    const double q13 = rotation_camera_to_world(0, 2);
    const double q21 = rotation_camera_to_world(1, 0);
    const double q22 = rotation_camera_to_world(1, 1);
    const double q23 = rotation_camera_to_world(1, 2);
    const double q33 = rotation_camera_to_world(2, 2);

    const double sin_beta = std::clamp(-q23, -1.0, 1.0);
    const double beta = std::asin(sin_beta);

    double alpha = 0.0;
    double gamma = 0.0;

    if (std::abs(std::cos(beta)) > 1e-6)
    {
        alpha = std::atan2(q13, q33);
        gamma = std::atan2(q21, q22);
    }
    else
    {
        gamma = 0.0;

        if (sin_beta > 0.0)
        {
            alpha = std::atan2(q12, q11);
        }
        else
        {
            alpha = std::atan2(-q12, q11);
        }
    }

    result.yaw = alpha * 180.0 / CV_PI;
    result.roll = beta * 180.0 / CV_PI;
    result.pitch = gamma * 180.0 / CV_PI;

    result.translation = translation_vector;
    result.depth = translation_vector[2];
    result.distance = cv::norm(translation_vector);

    return true;
}

static bool get_valid_keypoints(
    const std::vector<cv::Point3f>& keypoints,
    const cv::Size& image_size,
    std::vector<cv::Point>& points)
{
    points.clear();

    if (keypoints.size() != 4)
    {
        return false;
    }

    for (const auto& keypoint : keypoints)
    {
        if (!std::isfinite(keypoint.x) ||
            !std::isfinite(keypoint.y))
        {
            return false;
        }

        const cv::Point point(
            cvRound(keypoint.x),
            cvRound(keypoint.y)
        );

        if (point.x < 0 ||
            point.y < 0 ||
            point.x >= image_size.width ||
            point.y >= image_size.height)
        {
            return false;
        }

        points.push_back(point);
    }

    double min_distance = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        for (std::size_t j = i + 1; j < points.size(); ++j)
        {
            min_distance = std::min(
                min_distance,
                cv::norm(points[i] - points[j])
            );
        }
    }

    if (min_distance < 5.0)
    {
        return false;
    }

    return std::abs(cv::contourArea(points)) >= 20.0;
}

static std::vector<cv::Point> order_keypoints_for_display(
    const std::vector<cv::Point>& points)
{
    if (points.size() != 4)
    {
        return points;
    }

    std::vector<cv::Point> ordered_points = points;

    std::sort(
        ordered_points.begin(),
        ordered_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            if (a.y != b.y)
            {
                return a.y < b.y;
            }

            return a.x < b.x;
        }
    );

    std::vector<cv::Point> top_points(
        ordered_points.begin(),
        ordered_points.begin() + 2
    );

    std::vector<cv::Point> bottom_points(
        ordered_points.begin() + 2,
        ordered_points.end()
    );

    std::sort(
        top_points.begin(),
        top_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            return a.x < b.x;
        }
    );

    std::sort(
        bottom_points.begin(),
        bottom_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            return a.x > b.x;
        }
    );

    return {
        top_points[0],
        top_points[1],
        bottom_points[0],
        bottom_points[1]
    };
}

static bool construct_armor_corners(
    const std::vector<cv::Point>& raw_points,
    std::vector<cv::Point>& armor_corners)
{
    armor_corners.clear();

    if (raw_points.size() != 4)
    {
        return false;
    }

    std::vector<cv::Point> sorted_points = raw_points;

    std::sort(
        sorted_points.begin(),
        sorted_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            if (a.y != b.y)
            {
                return a.y < b.y;
            }

            return a.x < b.x;
        }
    );

    std::vector<cv::Point> top_points(
        sorted_points.begin(),
        sorted_points.begin() + 2
    );

    std::vector<cv::Point> bottom_points(
        sorted_points.begin() + 2,
        sorted_points.end()
    );

    std::sort(
        top_points.begin(),
        top_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            return a.x < b.x;
        }
    );

    std::sort(
        bottom_points.begin(),
        bottom_points.end(),
        [](const cv::Point& a, const cv::Point& b)
        {
            return a.x < b.x;
        }
    );

    const double top_width = cv::norm(top_points[1] - top_points[0]);
    const double bottom_width = cv::norm(
        bottom_points[1] - bottom_points[0]
    );

    const double pixel_width = (top_width + bottom_width) / 2.0;

    if (!std::isfinite(pixel_width) || pixel_width < 5.0)
    {
        return false;
    }

    const int offset = cvRound(pixel_width * 30.0 / 130.0);

    armor_corners = {
        cv::Point(top_points[0].x, top_points[0].y - offset),
        cv::Point(top_points[1].x, top_points[1].y - offset),
        cv::Point(bottom_points[1].x, bottom_points[1].y + offset),
        cv::Point(bottom_points[0].x, bottom_points[0].y + offset)
    };

    return true;
}

int main()
{
    Camera camera;

    if (!camera.initialize())
    {
        std::cerr << "Camera initialization failed: "
                  << camera.last_error() << '\n';
        return EXIT_FAILURE;
    }

    if (!camera.enumerate_devices())
    {
        std::cerr << "Camera enumeration failed: "
                  << camera.last_error() << '\n';
        camera.finalize();
        return EXIT_FAILURE;
    }

    if (camera.device_count() <= 0)
    {
        std::cerr << "No camera device found.\n";
        camera.finalize();
        return EXIT_FAILURE;
    }

    if (!camera.create_handle(0))
    {
        std::cerr << "Camera handle creation failed: "
                  << camera.last_error() << '\n';
        camera.finalize();
        return EXIT_FAILURE;
    }

    if (!camera.open())
    {
        std::cerr << "Camera open failed: "
                  << camera.last_error() << '\n';
        camera.finalize();
        return EXIT_FAILURE;
    }

    if (!camera.set_camera_parameters())
    {
        std::cerr << "Warning: camera parameter configuration failed: "
                  << camera.last_error() << '\n';
    }

    if (!camera.start_grabbing())
    {
        std::cerr << "Camera start grabbing failed: "
                  << camera.last_error() << '\n';
        camera.close();
        camera.finalize();
        return EXIT_FAILURE;
    }

    std::unique_ptr<YoloVino::Yolov5fourpointVino> yolo;

    try
    {
        auto logger = std::make_unique<YoloVino::YoloVinoLogger>(
            "vino_task/src/YoloVino/config/yolov5fourpoint_vino_config.yaml"
        );

        logger->set_info_level(
            YoloVino::LoggerInfoLevel::basic_info
        );

        yolo = std::make_unique<YoloVino::Yolov5fourpointVino>(
            std::move(logger)
        );
    }
    catch (const std::exception& exception)
    {
        std::cerr << "YOLO initialization failed: "
                  << exception.what() << '\n';
        camera.stop_grabbing();
        camera.close();
        camera.finalize();
        return EXIT_FAILURE;
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    cv::FileStorage calibration(
        "config/camera_calibration.yml",
        cv::FileStorage::READ
    );

    if (!calibration.isOpened())
    {
        std::cerr << "Cannot open config/camera_calibration.yml\n";
        camera.stop_grabbing();
        camera.close();
        camera.finalize();
        return EXIT_FAILURE;
    }

    calibration["camera_matrix"] >> camera_matrix;
    calibration["dist_coeffs"] >> dist_coeffs;
    calibration.release();

    if (camera_matrix.empty() || dist_coeffs.empty())
    {
        std::cerr << "Camera calibration data is missing.\n";
        camera.stop_grabbing();
        camera.close();
        camera.finalize();
        return EXIT_FAILURE;
    }

    if (camera_matrix.rows != 3 || camera_matrix.cols != 3)
    {
        std::cerr << "Invalid camera_matrix dimensions.\n";
        camera.stop_grabbing();
        camera.close();
        camera.finalize();
        return EXIT_FAILURE;
    }

    std::filesystem::create_directories("recordings");

    cv::VideoWriter writer;
    bool recording = false;
    unsigned long long frame_count = 0;

    cv::namedWindow("YOLO Armor Detection", cv::WINDOW_NORMAL);

    while (true)
    {
        cv::Mat image;

        if (!camera.get_frame(image) || image.empty())
        {
            std::cerr << "Failed to acquire camera frame: "
                      << camera.last_error() << '\n';

            if (cv::waitKey(1) == 27)
            {
                break;
            }

            continue;
        }

        cv::Mat inference_image;

        if (image.type() == CV_8UC3)
        {
            inference_image = image;
        }
        else if (image.type() == CV_8UC1)
        {
            cv::cvtColor(
                image,
                inference_image,
                cv::COLOR_GRAY2BGR
            );
        }
        else if (image.type() == CV_8UC4)
        {
            cv::cvtColor(
                image,
                inference_image,
                cv::COLOR_BGRA2BGR
            );
        }
        else
        {
            std::cerr << "Unsupported camera image format. Type: "
                      << image.type()
                      << ", channels: " << image.channels()
                      << ", depth: " << image.depth() << '\n';
            continue;
        }

        const cv::Rect full_roi(
            0,
            0,
            inference_image.cols,
            inference_image.rows
        );

        std::vector<YoloVino::NNDetectData> detections;

        try
        {
            detections = yolo->safe_predict(
                inference_image,
                full_roi
            );
        }
        catch (const std::exception& exception)
        {
            std::cerr << "YOLO inference failed: "
                      << exception.what() << '\n';
            continue;
        }

        ++frame_count;

        cv::Mat display_image = inference_image.clone();

        const cv::Rect image_bounds(
            0,
            0,
            display_image.cols,
            display_image.rows
        );

        for (const auto& detection : detections)
        {
            const cv::Rect bounds = detection.rect & image_bounds;

            if (bounds.area() <= 0)
            {
                continue;
            }

            cv::rectangle(
                display_image,
                bounds,
                cv::Scalar(0, 140, 255),
                2
            );

            std::ostringstream label;
            label << "ID:" << detection.class_id
                  << " Conf:" << std::fixed
                  << std::setprecision(2)
                  << detection.confidence;

            cv::putText(
                display_image,
                label.str(),
                cv::Point(bounds.x, std::max(20, bounds.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 140, 255),
                2
            );

            std::vector<cv::Point> points;

            const bool valid_keypoints = get_valid_keypoints(
                detection.keypoints,
                display_image.size(),
                points
            );

            if (!valid_keypoints)
            {
                cv::putText(
                    display_image,
                    "Invalid keypoints",
                    cv::Point(
                        bounds.x,
                        std::min(
                            display_image.rows - 10,
                            bounds.y + bounds.height + 20
                        )
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 255),
                    1
                );
                continue;
            }

            std::vector<cv::Point> armor_corners;

            if (!construct_armor_corners(points, armor_corners))
            {
                continue;
            }

            bool corners_in_bounds = true;

            for (const auto& point : armor_corners)
            {
                if (!image_bounds.contains(point))
                {
                    corners_in_bounds = false;
                    break;
                }
            }

            if (!corners_in_bounds)
            {
                continue;
            }

            for (const auto& point : points)
            {
                cv::circle(
                    display_image,
                    point,
                    4,
                    cv::Scalar(0, 0, 255),
                    cv::FILLED,
                    cv::LINE_AA
                );
            }

            cv::polylines(
                display_image,
                armor_corners,
                true,
                cv::Scalar(255, 0, 255),
                2,
                cv::LINE_AA
            );

            for (std::size_t i = 0; i < armor_corners.size(); ++i)
            {
                cv::circle(
                    display_image,
                    armor_corners[i],
                    5,
                    cv::Scalar(0, 255, 0),
                    cv::FILLED,
                    cv::LINE_AA
                );

                cv::putText(
                    display_image,
                    std::to_string(i),
                    armor_corners[i] + cv::Point(6, -6),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 255),
                    2,
                    cv::LINE_AA
                );
            }

            std::vector<cv::Point3f> armor_image_points;
            armor_image_points.reserve(4);

            for (const auto& point : armor_corners)
            {
                armor_image_points.emplace_back(
                    static_cast<float>(point.x),
                    static_cast<float>(point.y),
                    1.0F
                );
            }

            PnPResult pose;

            constexpr double armor_width = 130.0;
            constexpr double armor_height = 130.0;

            if (solve_armor_pose(
                    armor_image_points,
                    armor_width,
                    armor_height,
                    camera_matrix,
                    dist_coeffs,
                    pose))
            {
                std::ostringstream pose_text;
                pose_text << std::fixed << std::setprecision(1)
                          << "Yaw " << pose.yaw << " deg"
                          << " Roll " << pose.roll << " deg"
                          << " Pitch " << pose.pitch << " deg";

                std::ostringstream distance_text;
                distance_text << std::fixed << std::setprecision(1)
                              << "Distance " << pose.distance << " mm"
                              << "  Z " << pose.depth << " mm";

                cv::putText(
                    display_image,
                    pose_text.str(),
                    cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );

                cv::putText(
                    display_image,
                    distance_text.str(),
                    cv::Point(20, 65),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );
            }

            if (frame_count == 1 || frame_count % 30 == 0)
            {
                std::cout << "\n[Detection] frame=" << frame_count
                          << " class=" << detection.class_id
                          << " confidence=" << detection.confidence
                          << " box=(" << bounds.x << ","
                          << bounds.y << ","
                          << bounds.width << ","
                          << bounds.height << ")\n";

                for (std::size_t i = 0;
                     i < detection.keypoints.size();
                     ++i)
                {
                    const auto& point = detection.keypoints[i];

                    std::cout << "  raw_keypoint[" << i << "] = ("
                              << point.x << ", "
                              << point.y << "), confidence="
                              << point.z << '\n';
                }

                std::cout << "  PnP input uses constructed armor corners.\n";
            }
        }

        if (recording && !writer.isOpened())
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t timestamp =
                std::chrono::system_clock::to_time_t(now);

            std::tm local_time{};

#ifdef _WIN32
            localtime_s(&local_time, &timestamp);
#else
            localtime_r(&timestamp, &local_time);
#endif

            std::ostringstream filename;
            filename << "recordings/recording_"
                     << std::put_time(&local_time, "%Y%m%d_%H%M%S")
                     << ".avi";

            writer.open(
                filename.str(),
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                30.0,
                display_image.size(),
                true
            );

            if (!writer.isOpened())
            {
                std::cerr << "Failed to open video writer.\n";
                recording = false;
            }
        }

        if (recording && writer.isOpened())
        {
            cv::putText(
                display_image,
                "REC",
                cv::Point(20, 100),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                2
            );

            writer.write(display_image);
        }

        cv::imshow("YOLO Armor Detection", display_image);

        const int key = cv::waitKey(1);

        if (key == 27)
        {
            break;
        }

        if (key == 'r' || key == 'R')
        {
            recording = !recording;

            if (!recording && writer.isOpened())
            {
                writer.release();
            }

            std::cout << (recording
                ? "Recording started.\n"
                : "Recording stopped.\n");
        }
    }

    if (writer.isOpened())
    {
        writer.release();
    }

    cv::destroyAllWindows();

    camera.stop_grabbing();
    camera.close();
    camera.finalize();

    return EXIT_SUCCESS;
}

