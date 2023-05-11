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
#define FLANN_INDEX_KDTREE 1

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

static vector<float> calibration_to_depth_camera_dist_coeffs(const k4a::calibration& cal) {
    const k4a_calibration_intrinsic_parameters_t::_param& i = cal.depth_camera_calibration.intrinsics.parameters.param;
    return { i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6 };
}

static cv::Matx33f calibration_to_depth_camera_matrix(const k4a::calibration& cal) {
    const k4a_calibration_intrinsic_parameters_t::_param& i = cal.depth_camera_calibration.intrinsics.parameters.param;
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

k4a::image transform_depth_to_color_master(k4a_device_t capturer, k4a_device_configuration_t main_config, k4a::capture image) {
    k4a_transformation_t transformation = NULL;
    k4a_calibration_t calibration = get_calib(capturer, main_config);
    transformation = k4a_transformation_create(&calibration);
    k4a_image_t transformed = NULL;
    k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, k4a_image_get_width_pixels(image.get_color_image().handle()), k4a_image_get_height_pixels(image.get_color_image().handle()), 0, &transformed);
    k4a_transformation_depth_image_to_color_camera(transformation, image.get_depth_image().handle(), transformed);
    k4a_transformation_destroy(transformation);
    return k4a::image(transformed);
}

k4a::image transform_depth_to_color_slave(k4a_device_t capturer, k4a_device_configuration_t secondary_config, k4a::capture image) {
    k4a_transformation_t transformation = NULL;
    k4a_calibration_t calibration = get_calib(capturer, secondary_config);
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
        tr.R, // output
        tr.t, // output
        cv::noArray(),
        cv::noArray(),
        cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL | cv::CALIB_CB_FAST_CHECK);
    cout << "Finished calibrating!\n";
    cout << "Got error of " << error << "\n";

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

static k4a::calibration construct_device_to_device_calibration(const k4a::calibration& main_cal,
    const k4a::calibration& secondary_cal,
    const Transformation& secondary_to_main) {
    k4a::calibration cal = secondary_cal;
    k4a_calibration_extrinsics_t& ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ex.rotation[i * 3 + j] = static_cast<float>(secondary_to_main.R(i, j));
        }
    }
    for (int i = 0; i < 3; ++i) {
        ex.translation[i] = static_cast<float>(secondary_to_main.t[i]);
    }
    cal.color_camera_calibration = main_cal.color_camera_calibration;
    return cal;
}

static Transformation get_depth_to_color_transformation_from_calibration(const k4a::calibration& cal) {
    const k4a_calibration_extrinsics_t& ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    Transformation tr;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tr.R(i, j) = ex.rotation[i * 3 + j];
        }
    }
    tr.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
    return tr;
}

static k4a::image create_depth_image_like(const k4a::image& im) {
    return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
        im.get_width_pixels(),
        im.get_height_pixels(),
        im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

static k4a::image create_color_image_like(const k4a::image& im) {
    return k4a::image::create(im.get_format(),
        im.get_width_pixels(),
        im.get_height_pixels(),
        im.get_stride_bytes());
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        exit(1);
    }
    vector<uint32_t> device_indices{ 0, 1 };
    int depth_threshold = 500;
    uint32_t count = k4a_device_get_installed_count();
    printf("number of cameras: %d", count);
    k4a_device_configuration_t main_config = get_master_config();
    k4a_device_configuration_t secondary_config = get_subordinate_config();
    MultiDeviceCapturer capturer(device_indices, 8000, 1);
    capturer.start_devices(main_config, secondary_config);
    vector<k4a::capture> background_captures = capturer.get_synchronized_captures(secondary_config, true);
    k4a::image color1 = background_captures[0].get_color_image();
    k4a::image depth1 = background_captures[0].get_depth_image();
    k4a::image depth1_tr = transform_depth_to_color_master(capturer.get_master_device().handle(), main_config, background_captures[0]);
    k4a::image color2 = background_captures[1].get_color_image();
    k4a::image depth2 = background_captures[1].get_depth_image();
    k4a::image depth2_tr = transform_depth_to_color_slave(capturer.get_subordinate_device_by_index(0).handle(), secondary_config, background_captures[1]);
    cv::Mat image = color_to_opencv(color1);
    cv::Mat image1 = depth_to_opencv(depth1_tr);
    cv::Mat image2 = color_to_opencv(color2);
    cv::Mat image3 = depth_to_opencv(depth2_tr);
    Transformation tr_secondary_color_to_main_color;

    if (argv[1][0] == '1') {
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

    k4a::calibration main_calibration =
        capturer.get_master_device().get_calibration(main_config.depth_mode,
            main_config.color_resolution);

    k4a::calibration secondary_calibration =
        capturer.get_subordinate_device_by_index(0).get_calibration(secondary_config.depth_mode,
            secondary_config.color_resolution);

    Transformation tr_secondary_depth_to_secondary_color = get_depth_to_color_transformation_from_calibration(
        secondary_calibration);

    Transformation tr_secondary_depth_to_main_color = tr_secondary_color_to_main_color.compose_with(
        tr_secondary_depth_to_secondary_color);

    k4a::calibration secondary_depth_to_main_color_cal =
        construct_device_to_device_calibration(main_calibration,
            secondary_calibration,
            tr_secondary_depth_to_main_color);
    k4a::transformation secondary_depth_to_main_color(secondary_depth_to_main_color_cal);

    k4a::calibration secondary_color_to_main_color_cal =
        construct_device_to_device_calibration(main_calibration,
            secondary_calibration,
            tr_secondary_color_to_main_color);

    k4a::transformation main_depth_to_main_color(main_calibration);
    k4a::image main_depth_in_main_color = create_depth_image_like(color1);
    main_depth_to_main_color.depth_image_to_color_camera(depth1, &main_depth_in_main_color);
    cv::Mat cv_main_depth_in_main_color = depth_to_opencv(main_depth_in_main_color);

    k4a::image secondary_depth_in_main_color = create_depth_image_like(color1);
    secondary_depth_to_main_color.depth_image_to_color_camera(depth2,
        &secondary_depth_in_main_color);
    cv::Mat cv_secondary_depth_in_main_color = depth_to_opencv(secondary_depth_in_main_color);

    cv::Mat main_valid_mask = cv_main_depth_in_main_color != 0;
    cv::Mat secondary_valid_mask = cv_secondary_depth_in_main_color != 0;

    cv::Mat within_threshold_range_1 = (image1 != 0) &
        (image1 < depth_threshold);

    cv::Mat within_threshold_range_2 = (image3 != 0) &
        (image3 < depth_threshold);

    cv::Mat outp1, outp2, outp3, outp4;

    image.copyTo(outp1, within_threshold_range_1);
    cv::imwrite("color1.jpg", outp1);
    image2.copyTo(outp2, within_threshold_range_2);
    cv::imwrite("color2.jpg", outp2);
    image1.copyTo(outp3, within_threshold_range_1);
    cv::imwrite("depth1.png", outp3);
    image3.copyTo(outp4, within_threshold_range_2);
    cv::imwrite("depth2.png", outp4);

    //cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    //std::vector<cv::KeyPoint> keypoints1, keypoints2;
    //cv::Mat descriptors1, descriptors2;
    //siftPtr->detectAndCompute(image, cv::noArray(), keypoints1, descriptors1);
    //siftPtr->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    //cv::Mat res1, res2;
    //cv::drawKeypoints(image, keypoints1, res1);
    //cv::drawKeypoints(image2, keypoints2, res2);
    //cv::imwrite("sift_test.jpg", res1);
    //cv::imwrite("sift_test2.jpg", res2);

    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //std::vector< std::vector<cv::DMatch> > knn_matches;
    //matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    //std::vector<cv::Point2f> good_keys1, good_keys2;

    //const float ratio_thresh = 0.7f;
    //std::vector<cv::DMatch> good_matches;
    //for (size_t i = 0; i < knn_matches.size(); i++)
    //{
    //    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //    {
    //        good_matches.push_back(knn_matches[i][0]);
    //        good_keys1.push_back(keypoints1[knn_matches.data()[i][0].queryIdx].pt);
    //        good_keys2.push_back(keypoints2[knn_matches.data()[i][1].trainIdx].pt);
    //    }
    //}

    //cv::Mat img_matches;
    //drawMatches(image, keypoints1, image2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
    //    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //cv::imwrite("img_matches.jpg", img_matches);



    //cv::Mat fund = cv::findFundamentalMat(good_keys1, good_keys2, cv::FM_RANSAC);

    cv::Matx33f matr1 = calibration_to_color_camera_matrix(main_calibration);
    std::vector<float> dist1 = calibration_to_color_camera_dist_coeffs(main_calibration);
    cv::Matx33f matr2 = calibration_to_color_camera_matrix(secondary_calibration);
    std::vector<float> dist2 = calibration_to_color_camera_dist_coeffs(secondary_calibration);
    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Mat map1, map2, map3, map4, map5, map6, map7, map8;
    cv::Mat routp1, routp2, routp3, routp4;

    cv::stereoRectify(matr2, dist2, matr1, dist1, outp1.size(), tr_secondary_color_to_main_color.R, tr_secondary_color_to_main_color.t, R1, R2, P1, P2, Q);

    cv::initUndistortRectifyMap(calibration_to_color_camera_matrix(main_calibration), calibration_to_color_camera_dist_coeffs(main_calibration), R2, cv::noArray(), outp1.size(), CV_32FC1, map1, map2);
    cv::initUndistortRectifyMap(calibration_to_color_camera_matrix(secondary_calibration), calibration_to_color_camera_dist_coeffs(secondary_calibration), R1, cv::noArray(), outp2.size(), CV_32FC1, map3, map4);
    cv::initUndistortRectifyMap(calibration_to_depth_camera_matrix(main_calibration), calibration_to_depth_camera_dist_coeffs(main_calibration), R2, cv::noArray(), outp3.size(), CV_32FC1, map5, map6);
    cv::initUndistortRectifyMap(calibration_to_depth_camera_matrix(secondary_calibration), calibration_to_depth_camera_dist_coeffs(secondary_calibration), R1, cv::noArray(), outp4.size(), CV_32FC1, map7, map8);

    cv::remap(outp1, routp1, map1, map2, cv::INTER_LINEAR);
    cv::remap(outp2, routp2, map3, map4, cv::INTER_LINEAR);
    cv::remap(outp3, routp3, map5, map6, cv::INTER_LINEAR);
    cv::remap(outp4, routp4, map7, map8, cv::INTER_LINEAR);

    cv::imwrite("rectified1.jpg", routp1);
    cv::imwrite("rectified2.jpg", routp2);
    cv::imwrite("rectified3.png", routp3);
    cv::imwrite("rectified4.png", routp4);

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

    ofstream myfile("intrinsic1.json");
    if (myfile.is_open()) {
        myfile << "{\"intrinsic_matrix\" : [";
        for (int count = 0; count < main_calibration.depth_camera_calibration.intrinsics.parameter_count; count++) {
            if (count == (main_calibration.depth_camera_calibration.intrinsics.parameter_count - 1)) {
                myfile << main_calibration.depth_camera_calibration.intrinsics.parameters.v[count] << "]}";
                continue;
            }
            myfile << main_calibration.depth_camera_calibration.intrinsics.parameters.v[count] << ", ";
        }
        myfile.close();
    }
    ofstream myfile2("intrinsic2.json");
    if (myfile2.is_open()) {
        myfile2 << "{\"intrinsic_matrix\" : [";
        for (int count = 0; count < secondary_calibration.depth_camera_calibration.intrinsics.parameter_count; count++) {
            if (count == (secondary_calibration.depth_camera_calibration.intrinsics.parameter_count - 1)) {
                myfile2 << secondary_calibration.depth_camera_calibration.intrinsics.parameters.v[count] << "]}";
                continue;
            }
            myfile2 << secondary_calibration.depth_camera_calibration.intrinsics.parameters.v[count] << ", ";
        }
        myfile2.close();
    }
    return 0;
}