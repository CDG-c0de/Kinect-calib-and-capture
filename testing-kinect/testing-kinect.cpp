#include <k4a/k4a.h>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;
#include <vector>
#include <string>
#include <chrono>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using std::vector;
using std::cerr;
using std::cout;
using std::endl;

#include "MultiDeviceCapturer.h"
#include "transformation.h"
#include "json.hpp"
using json = nlohmann::json;

#define MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC 160
#define CHESSBOARD_SQUARE_SIZE 24

static cv::Mat depth_to_opencv(const k4a::image& im) {
    return cv::Mat(im.get_height_pixels(),
        im.get_width_pixels(),
        CV_16U,
        (void*)im.get_buffer(),
        static_cast<size_t>(im.get_stride_bytes()));
}

static cv::Mat color_to_opencv(const k4a::image& im) {
    cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void*)im.get_buffer());
    cv::Mat cv_image_no_alpha;
    cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
    return cv_image_no_alpha;
}

static k4a_device_configuration_t get_default_config() {
    k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    camera_config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    camera_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
    camera_config.subordinate_delay_off_master_usec = 0;
    camera_config.synchronized_images_only = true;
    return camera_config;
}

static k4a_device_configuration_t get_master_config() {
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
    camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
    camera_config.synchronized_images_only = true;
    return camera_config;
}

static k4a_device_configuration_t get_subordinate_config() {
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
    camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
    return camera_config;
}

static vector<float> calibration_to_color_camera_dist_coeffs(const k4a::calibration& cal) {
    const k4a_calibration_intrinsic_parameters_t::_param& i = cal.color_camera_calibration.intrinsics.parameters.param;
    return { i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6 };
}

static cv::Matx33f calibration_to_color_camera_matrix(const k4a::calibration& cal) {
    const k4a_calibration_intrinsic_parameters_t::_param& i = cal.color_camera_calibration.intrinsics.parameters.param;
    cv::Matx33f camera_matrix = cv::Matx33f::eye();
    camera_matrix(0, 0) = i.fx;
    camera_matrix(1, 1) = i.fy;
    camera_matrix(0, 2) = i.cx;
    camera_matrix(1, 2) = i.cy;
    return camera_matrix;
}

k4a_calibration_t get_calib(k4a_device_t capturer, k4a_device_configuration_t config) {
    k4a_calibration_t calibration;
    k4a_device_get_calibration(capturer, config.depth_mode, config.color_resolution, &calibration);
    return calibration;
}

k4a::image transform_depth_to_color(k4a_device_t capturer, k4a_device_configuration_t main_config, k4a::capture image) {
    k4a_transformation_t transformation = NULL;
    k4a_calibration_t calibration = get_calib(capturer, main_config);
    transformation = k4a_transformation_create(&calibration);
    k4a_image_t transformed = NULL;
    k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, k4a_image_get_width_pixels(image.get_color_image().handle()), k4a_image_get_height_pixels(image.get_color_image().handle()), 0, &transformed);
    k4a_transformation_depth_image_to_color_camera(transformation, image.get_depth_image().handle(), transformed);
    k4a_transformation_destroy(transformation);
    return k4a::image(transformed);
}

bool find_chessboard_corners_helper(const cv::Mat& main_color_image,
    const cv::Mat& secondary_color_image,
    const cv::Size& chessboard_pattern,
    vector<cv::Point2f>& main_chessboard_corners,
    vector<cv::Point2f>& secondary_chessboard_corners) {

    bool found_chessboard_main = cv::findChessboardCorners(main_color_image,
        chessboard_pattern,
        main_chessboard_corners);
    bool found_chessboard_secondary = cv::findChessboardCorners(secondary_color_image,
        chessboard_pattern,
        secondary_chessboard_corners);

    if (!found_chessboard_main || !found_chessboard_secondary) {
        if (found_chessboard_main) {
            cout << "Could not find the chessboard corners in the secondary image. Trying again...\n";
        }
        else if (found_chessboard_secondary) {
            cout << "Could not find the chessboard corners in the main image. Trying again...\n";
        }
        else {
            cout << "Could not find the chessboard corners in either image. Trying again...\n";
        }
        return false;
    }

    cv::Vec2f main_image_corners_vec = main_chessboard_corners.back() - main_chessboard_corners.front();
    cv::Vec2f secondary_image_corners_vec = secondary_chessboard_corners.back() - secondary_chessboard_corners.front();
    if (main_image_corners_vec.dot(secondary_image_corners_vec) <= 0.0) {
        std::reverse(secondary_chessboard_corners.begin(), secondary_chessboard_corners.end());
    }
    return true;
}

Transformation stereo_calibration(const k4a::calibration& main_calib,
    const k4a::calibration& secondary_calib,
    const vector<vector<cv::Point2f>>& main_chessboard_corners_list,
    const vector<vector<cv::Point2f>>& secondary_chessboard_corners_list,
    const cv::Size& image_size,
    const cv::Size& chessboard_pattern,
    float chessboard_square_length) {

    vector<cv::Point3f> chessboard_corners_world;
    for (int h = 0; h < chessboard_pattern.height; ++h) {
        for (int w = 0; w < chessboard_pattern.width; ++w) {
            chessboard_corners_world.emplace_back(
                cv::Point3f{ w * chessboard_square_length, h * chessboard_square_length, 0.0 });
        }
    }

    vector<vector<cv::Point3f>> chessboard_corners_world_nested_for_cv(main_chessboard_corners_list.size(),
        chessboard_corners_world);

    cv::Matx33f main_camera_matrix = calibration_to_color_camera_matrix(main_calib);
    cv::Matx33f secondary_camera_matrix = calibration_to_color_camera_matrix(secondary_calib);
    vector<float> main_dist_coeff = calibration_to_color_camera_dist_coeffs(main_calib);
    vector<float> secondary_dist_coeff = calibration_to_color_camera_dist_coeffs(secondary_calib);

    Transformation tr;
    double error = cv::stereoCalibrate(chessboard_corners_world_nested_for_cv,
        secondary_chessboard_corners_list,
        main_chessboard_corners_list,
        secondary_camera_matrix,
        secondary_dist_coeff,
        main_camera_matrix,
        main_dist_coeff,
        image_size,
        tr.R,
        tr.t,
        cv::noArray(),
        cv::noArray(),
        cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL | cv::CALIB_CB_FAST_CHECK);
    cout << "Finished calibrating!\n";
    cout << "Got error of " << error << "\n";
    cout << tr.R << std::endl;
    cout << tr.t << std::endl;
    cout << tr.to_homogeneous() << std::endl;
    return tr;
}

static Transformation calibrate_devices(MultiDeviceCapturer& capturer,
    const k4a_device_configuration_t& main_config,
    const k4a_device_configuration_t& secondary_config,
    const cv::Size& chessboard_pattern,
    float chessboard_square_length,
    double calibration_timeout) {

    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,
        main_config.color_resolution);

    k4a::calibration secondary_calibration =
        capturer.get_subordinate_device_by_index(0).get_calibration(secondary_config.depth_mode,
            secondary_config.color_resolution);
    vector<vector<cv::Point2f>> main_chessboard_corners_list;
    vector<vector<cv::Point2f>> secondary_chessboard_corners_list;
    std::cout << "Please proceed to calibrate the cameras, use the chessboard from a different angle in each shot." << std::endl;
    std::cout << "Ensure the chessboard is visible for both cameras at the same time for each shot." << std::endl;
    while (1) {
        std::cout << "Amount of images to go: " << 50 - main_chessboard_corners_list.size() << std::endl;
        system("pause");
        vector<k4a::capture> captures = capturer.get_synchronized_captures(secondary_config);
        k4a::capture& main_capture = captures[0];
        k4a::capture& secondary_capture = captures[1];
        k4a::image main_color_image = main_capture.get_color_image();
        k4a::image secondary_color_image = secondary_capture.get_color_image();
        cv::Mat cv_main_color_image = color_to_opencv(main_color_image);
        cv::Mat cv_secondary_color_image = color_to_opencv(secondary_color_image);

        vector<cv::Point2f> main_chessboard_corners;
        vector<cv::Point2f> secondary_chessboard_corners;
        bool got_corners = find_chessboard_corners_helper(cv_main_color_image,
            cv_secondary_color_image,
            chessboard_pattern,
            main_chessboard_corners,
            secondary_chessboard_corners);
        if (got_corners) {
            main_chessboard_corners_list.emplace_back(main_chessboard_corners);
            secondary_chessboard_corners_list.emplace_back(secondary_chessboard_corners);
            cv::drawChessboardCorners(cv_main_color_image, chessboard_pattern, main_chessboard_corners, true);
            cv::drawChessboardCorners(cv_secondary_color_image, chessboard_pattern, secondary_chessboard_corners, true);
        }

        cv::imshow("Chessboard view from main camera", cv_main_color_image);
        cv::waitKey(1);
        cv::imshow("Chessboard view from secondary camera", cv_secondary_color_image);
        cv::waitKey(1);

        if (main_chessboard_corners_list.size() >= 50) {
            cout << "Calculating calibration..." << endl;
            return stereo_calibration(main_calibration,
                secondary_calibration,
                main_chessboard_corners_list,
                secondary_chessboard_corners_list,
                cv_main_color_image.size(),
                chessboard_pattern,
                chessboard_square_length);
        }
    }
    std::cerr << "Calibration timed out !\n ";
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        exit(1);
    }
    uint32_t count = k4a_device_get_installed_count();
    if (count < 1) {
        std::cout << "connect at least one device" << std::endl;
        exit(1);
    }
    vector<uint32_t> device_indices{};
    for (int i = 0; i < count; i++) {
        device_indices.push_back(i);
    }
    int depth_threshold = 500;
    printf("number of cameras: %d", count);
    k4a_device_configuration_t main_config = get_master_config();
    k4a_device_configuration_t secondary_config = get_subordinate_config();
    MultiDeviceCapturer capturer(device_indices, 8000, 1);
    if (count == 1) {
        capturer.start_devices(get_default_config(), secondary_config);
    } else {
        capturer.start_devices(main_config, secondary_config);
    }
    vector<k4a::capture> background_captures = capturer.get_synchronized_captures(secondary_config, true);
    vector<cv::Mat> color_images;
    vector<cv::Mat> depth_images;
    vector<k4a::image> temp_depths;
    for (int i = 0; i < count; i++) {
        color_images.push_back(color_to_opencv(background_captures[i].get_color_image()));
        if (i == 0) {
            temp_depths.push_back(transform_depth_to_color(capturer.get_master_device().handle(), main_config, background_captures[i]));
            depth_images.push_back(depth_to_opencv(temp_depths[i]));
        } else {
            temp_depths.push_back(transform_depth_to_color(capturer.get_subordinate_device_by_index(i - 1).handle(), secondary_config, background_captures[i]));
            depth_images.push_back(depth_to_opencv(temp_depths[i]));
        }
        std::stringstream n;
        n << "col" << i << ".jpg";
        cv::imwrite(n.str(), color_images[i]);
        n.str(std::string());
        n << "dep" << i << ".png";
        cv::imwrite(n.str(), depth_images[i]);
        n.str(std::string());
    }
    Transformation tr_secondary_color_to_main_color;

    if (argv[1][0] == '1') {
        if (count != 2) {
            std::cout << "Stereo calibration only supports 2 cameras" << std::endl;
            exit(1);
        }
        tr_secondary_color_to_main_color = calibrate_devices(capturer,
            main_config,
            secondary_config,
            cv::Size(9, 6),
            CHESSBOARD_SQUARE_SIZE,
            60.00);
    }
    else {
        std::ifstream f("extrinsic.json");
        json data = json::parse(f);
        for (int i = 0; i < data["rotation_matrix"].size(); i++) {
            tr_secondary_color_to_main_color.R.val[i] = data["rotation_matrix"].at(i);
        }
        for (int j = 0; j < data["translation_matrix"].size(); j++) {
            tr_secondary_color_to_main_color.t.val[j] = data["translation_matrix"].at(j);
        }
    }

    std::vector<k4a::calibration> cals;
    for (int i = 0; i < count; i++) {
        if (i == 0) {
            cals.push_back(capturer.get_master_device().get_calibration(main_config.depth_mode, main_config.color_resolution));
        } else {
            cals.push_back(capturer.get_subordinate_device_by_index(i-1).get_calibration(secondary_config.depth_mode, secondary_config.color_resolution));
        }
    }

    for (int i = 0; i < count; i++) {
        cv::Mat temp, temp2;
        temp2 = (depth_images[i] != 0) & (depth_images[i] < depth_threshold);
        color_images[i].copyTo(temp, temp2);
        std::stringstream n;
        n << "color" << i << ".jpg";
        cv::imwrite(n.str(), temp);
        depth_images[i].copyTo(temp, temp2);
        n.str(std::string());
        n << "depth" << i << ".png";
        cv::imwrite(n.str(), temp);
        n.str(std::string());
    }

    cv::Matx44d homo;
    homo = tr_secondary_color_to_main_color.to_homogeneous();

    if (argv[1][0] == '1') {
        ofstream exfile("extrinsic.json");
        if (exfile.is_open()) {
            exfile << "{\"rotation_matrix\" : [";
            for (int count = 0; count < tr_secondary_color_to_main_color.R.channels; count++) {
                if (count == (tr_secondary_color_to_main_color.R.channels - 1)) {
                    exfile << tr_secondary_color_to_main_color.R.val[count] << "], ";
                    continue;
                }
                exfile << tr_secondary_color_to_main_color.R.val[count] << ", ";
            }
            exfile << "\"translation_matrix\" : [";
            for (int count = 0; count < tr_secondary_color_to_main_color.t.channels; count++) {
                if (count == (tr_secondary_color_to_main_color.t.channels - 1)) {
                    exfile << tr_secondary_color_to_main_color.t.val[count] << "], ";
                    continue;
                }
                exfile << tr_secondary_color_to_main_color.t.val[count] << ", ";
            }
            exfile << "\"homo_matrix\" : [";
            for (int count = 0; count < homo.channels; count++) {
                if (count == (homo.channels - 1)) {
                    exfile << homo.val[count] << "]}";
                    continue;
                }
                exfile << homo.val[count] << ", ";
            }
            exfile.close();
        }
    }
    for (int i = 0; i < count; i++) {
        std::stringstream n;
        n << "intrinsic" << i << ".json";
        ofstream myfile(n.str());
        if (myfile.is_open()) {
            myfile << "{\"intrinsic_matrix\" : [";
            for (int count = 0; count < cals[i].depth_camera_calibration.intrinsics.parameter_count; count++) {
                if (count == (cals[i].depth_camera_calibration.intrinsics.parameter_count - 1)) {
                    myfile << cals[i].depth_camera_calibration.intrinsics.parameters.v[count] << "]}";
                    continue;
                }
                myfile << cals[i].depth_camera_calibration.intrinsics.parameters.v[count] << ", ";
            }
            myfile.close();
        }
        n.str(std::string());
    }
    return 0;
}