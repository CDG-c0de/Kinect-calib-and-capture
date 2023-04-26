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

static cv::Mat depth_to_opencv(const k4a::image& im)
{
    return cv::Mat(im.get_height_pixels(),
        im.get_width_pixels(),
        CV_16U,
        (void*)im.get_buffer(),
        static_cast<size_t>(im.get_stride_bytes()));
}

static cv::Mat color_to_opencv(const k4a::image& im)
{
    cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void*)im.get_buffer());
    cv::Mat cv_image_no_alpha;
    cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
    return cv_image_no_alpha;
}

static k4a_device_configuration_t get_default_config()
{
    k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    camera_config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED; // No need for depth during calibration
    camera_config.camera_fps = K4A_FRAMES_PER_SECOND_15;     // Don't use all USB bandwidth
    camera_config.subordinate_delay_off_master_usec = 0;     // Must be zero for master
    camera_config.synchronized_images_only = true;
    return camera_config;
}

static k4a_device_configuration_t get_master_config()
{
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

    // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
    // sensor doesn't interfere with the other. To accomplish this the master depth image captures
    // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
    // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
    // images centered around the color image as closely as possible.
    camera_config.depth_delay_off_color_usec = -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
    camera_config.synchronized_images_only = true;
    return camera_config;
}

static k4a_device_configuration_t get_subordinate_config()
{
    k4a_device_configuration_t camera_config = get_default_config();
    camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

    // Two depth images should be seperated by MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
    // sensor doesn't interfere with the other. To accomplish this the master depth image captures
    // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) before the color image, and the subordinate camera captures its
    // depth image (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image. This gives us two depth
    // images centered around the color image as closely as possible.
    camera_config.depth_delay_off_color_usec = MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
    return camera_config;
}

static vector<float> calibration_to_color_camera_dist_coeffs(const k4a::calibration& cal)
{
    const k4a_calibration_intrinsic_parameters_t::_param& i = cal.color_camera_calibration.intrinsics.parameters.param;
    return { i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6 };
}

static cv::Matx33f calibration_to_color_camera_matrix(const k4a::calibration& cal)
{
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

    // Cover the failure cases where chessboards were not found in one or both images.
    if (!found_chessboard_main || !found_chessboard_secondary)
    {
        if (found_chessboard_main)
        {
            cout << "Could not find the chessboard corners in the secondary image. Trying again...\n";
        }
        // Likewise, if the chessboard was found in the secondary image, it was not found in the main image.
        else if (found_chessboard_secondary)
        {
            cout << "Could not find the chessboard corners in the main image. Trying again...\n";
        }
        // The only remaining case is the corners were in neither image.
        else
        {
            cout << "Could not find the chessboard corners in either image. Trying again...\n";
        }
        return false;
    }
    // Before we go on, there's a quick problem with calibration to address.  Because the chessboard looks the same when
    // rotated 180 degrees, it is possible that the chessboard corner finder may find the correct points, but in the
    // wrong order.

    // A visual:
    //        Image 1                  Image 2
    // .....................    .....................
    // .....................    .....................
    // .........xxxxx2......    .....xxxxx1..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........1xxxxx......    .....2xxxxx..........
    // .....................    .....................
    // .....................    .....................

    // The problem occurs when this case happens: the find_chessboard() function correctly identifies the points on the
    // chessboard (shown as 'x's) but the order of those points differs between images taken by the two cameras.
    // Specifically, the first point in the list of points found for the first image (1) is the *last* point in the list
    // of points found for the second image (2), though they correspond to the same physical point on the chessboard.

    // To avoid this problem, we can make the assumption that both of the cameras will be oriented in a similar manner
    // (e.g. turning one of the cameras upside down will break this assumption) and enforce that the vector between the
    // first and last points found in pixel space (which will be at opposite ends of the chessboard) are pointing the
    // same direction- so, the dot product of the two vectors is positive.

    cv::Vec2f main_image_corners_vec = main_chessboard_corners.back() - main_chessboard_corners.front();
    cv::Vec2f secondary_image_corners_vec = secondary_chessboard_corners.back() - secondary_chessboard_corners.front();
    if (main_image_corners_vec.dot(secondary_image_corners_vec) <= 0.0)
    {
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
    float chessboard_square_length)
{
    // We have points in each image that correspond to the corners that the findChessboardCorners function found.
    // However, we still need the points in 3 dimensions that these points correspond to. Because we are ultimately only
    // interested in find a transformation between two cameras, these points don't have to correspond to an external
    // "origin" point. The only important thing is that the relative distances between points are accurate. As a result,
    // we can simply make the first corresponding point (0, 0) and construct the remaining points based on that one. The
    // order of points inserted into the vector here matches the ordering of findChessboardCorners. The units of these
    // points are in millimeters, mostly because the depth provided by the depth cameras is also provided in
    // millimeters, which makes for easy comparison.
    vector<cv::Point3f> chessboard_corners_world;
    for (int h = 0; h < chessboard_pattern.height; ++h)
    {
        for (int w = 0; w < chessboard_pattern.width; ++w)
        {
            chessboard_corners_world.emplace_back(
                cv::Point3f{ w * chessboard_square_length, h * chessboard_square_length, 0.0 });
        }
    }

    // Calibrating the cameras requires a lot of data. OpenCV's stereoCalibrate function requires:
    // - a list of points in real 3d space that will be used to calibrate*
    // - a corresponding list of pixel coordinates as seen by the first camera*
    // - a corresponding list of pixel coordinates as seen by the second camera*
    // - the camera matrix of the first camera
    // - the distortion coefficients of the first camera
    // - the camera matrix of the second camera
    // - the distortion coefficients of the second camera
    // - the size (in pixels) of the images
    // - R: stereoCalibrate stores the rotation matrix from the first camera to the second here
    // - t: stereoCalibrate stores the translation vector from the first camera to the second here
    // - E: stereoCalibrate stores the essential matrix here (we don't use this)
    // - F: stereoCalibrate stores the fundamental matrix here (we don't use this)
    //
    // * note: OpenCV's stereoCalibrate actually requires as input an array of arrays of points for these arguments,
    // allowing a caller to provide multiple frames from the same camera with corresponding points. For example, if
    // extremely high precision was required, many images could be taken with each camera, and findChessboardCorners
    // applied to each of those images, and OpenCV can jointly solve for all of the pairs of corresponding images.
    // However, to keep things simple, we use only one image from each device to calibrate.  This is also why each of
    // the vectors of corners is placed into another vector.
    //
    // A function in OpenCV's calibration function also requires that these points be F32 types, so we use those.
    // However, OpenCV still provides doubles as output, strangely enough.
    vector<vector<cv::Point3f>> chessboard_corners_world_nested_for_cv(main_chessboard_corners_list.size(),
        chessboard_corners_world);

    cv::Matx33f main_camera_matrix = calibration_to_color_camera_matrix(main_calib);
    cv::Matx33f secondary_camera_matrix = calibration_to_color_camera_matrix(secondary_calib);
    vector<float> main_dist_coeff = calibration_to_color_camera_dist_coeffs(main_calib);
    vector<float> secondary_dist_coeff = calibration_to_color_camera_dist_coeffs(secondary_calib);

    // Finally, we'll actually calibrate the cameras.
    // Pass secondary first, then main, because we want a transform from secondary to main.
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
    //std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
    std::cout << "Please proceed to calibrate the cameras, use the chessboard from a different angle in each shot." << std::endl;
    std::cout << "Ensure the chessboard is visible for both cameras at the same time for each shot." << std::endl;
    while (1)
    {
        std::cout << "Amount of images to go: " << 50 - main_chessboard_corners_list.size() << std::endl;
        system("pause");
        vector<k4a::capture> captures = capturer.get_synchronized_captures(secondary_config);
        k4a::capture& main_capture = captures[0];
        k4a::capture& secondary_capture = captures[1];
        // get_color_image is guaranteed to be non-null because we use get_synchronized_captures for color
        // (get_synchronized_captures also offers a flag to use depth for the secondary camera instead of color).
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
        if (got_corners)
        {
            main_chessboard_corners_list.emplace_back(main_chessboard_corners);
            secondary_chessboard_corners_list.emplace_back(secondary_chessboard_corners);
            cv::drawChessboardCorners(cv_main_color_image, chessboard_pattern, main_chessboard_corners, true);
            cv::drawChessboardCorners(cv_secondary_color_image, chessboard_pattern, secondary_chessboard_corners, true);
        }

        cv::imshow("Chessboard view from main camera", cv_main_color_image);
        cv::waitKey(1);
        cv::imshow("Chessboard view from secondary camera", cv_secondary_color_image);
        cv::waitKey(1);

        // Get 20 frames before doing calibration.
        if (main_chessboard_corners_list.size() >= 50)
        {
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
    const Transformation& secondary_to_main)
{
    k4a::calibration cal = secondary_cal;
    k4a_calibration_extrinsics_t& ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            ex.rotation[i * 3 + j] = static_cast<float>(secondary_to_main.R(i, j));
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        ex.translation[i] = static_cast<float>(secondary_to_main.t[i]);
    }
    cal.color_camera_calibration = main_cal.color_camera_calibration;
    return cal;
}

static Transformation get_depth_to_color_transformation_from_calibration(const k4a::calibration& cal)
{
    const k4a_calibration_extrinsics_t& ex = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    Transformation tr;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            tr.R(i, j) = ex.rotation[i * 3 + j];
        }
    }
    tr.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
    return tr;
}

static k4a::image create_depth_image_like(const k4a::image& im)
{
    return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
        im.get_width_pixels(),
        im.get_height_pixels(),
        im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

static k4a::image create_color_image_like(const k4a::image& im)
{
    return k4a::image::create(im.get_format(),
        im.get_width_pixels(),
        im.get_height_pixels(),
        im.get_stride_bytes());
}

int main(int argc, char * argv[]) {
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

        //Perform calibration
        tr_secondary_color_to_main_color = calibrate_devices(capturer,
            main_config,
            secondary_config,
            cv::Size(9, 6),
            CHESSBOARD_SQUARE_SIZE,
            60.00);
    } else {
        std::cout << "testing" << std::endl;
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

    // Get the transformation from secondary depth to secondary color using its calibration object
    Transformation tr_secondary_depth_to_secondary_color = get_depth_to_color_transformation_from_calibration(
        secondary_calibration);

    // We now have the secondary depth to secondary color transform. We also have the transformation from the
    // secondary color perspective to the main color perspective from the calibration earlier. Now let's compose the
    // depth secondary -> color secondary, color secondary -> color main into depth secondary -> color main
    Transformation tr_secondary_depth_to_main_color = tr_secondary_color_to_main_color.compose_with(
        tr_secondary_depth_to_secondary_color);

    // Construct a new calibration object to transform from the secondary depth camera to the main color camera
    k4a::calibration secondary_depth_to_main_color_cal =
        construct_device_to_device_calibration(main_calibration,
            secondary_calibration,
            tr_secondary_depth_to_main_color);
    k4a::transformation secondary_depth_to_main_color(secondary_depth_to_main_color_cal);

    k4a::calibration secondary_color_to_main_color_cal =
        construct_device_to_device_calibration(main_calibration,
            secondary_calibration,
            tr_secondary_color_to_main_color);

    //Create images

    k4a::transformation main_depth_to_main_color(main_calibration);
    k4a::image main_depth_in_main_color = create_depth_image_like(color1);
    main_depth_to_main_color.depth_image_to_color_camera(depth1, &main_depth_in_main_color);
    cv::Mat cv_main_depth_in_main_color = depth_to_opencv(main_depth_in_main_color);

    // Get the depth image in the main color perspective
    k4a::image secondary_depth_in_main_color = create_depth_image_like(color1);
    secondary_depth_to_main_color.depth_image_to_color_camera(depth2,
        &secondary_depth_in_main_color);
    cv::Mat cv_secondary_depth_in_main_color = depth_to_opencv(secondary_depth_in_main_color);

    cv::Mat main_valid_mask = cv_main_depth_in_main_color != 0;
    cv::Mat secondary_valid_mask = cv_secondary_depth_in_main_color != 0;

    cv::Mat within_threshold_range_1 = (main_valid_mask & (cv_main_depth_in_main_color < depth_threshold)) |
        (~main_valid_mask & secondary_valid_mask &
            (cv_secondary_depth_in_main_color < depth_threshold));

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
                    exfile << tr_secondary_color_to_main_color.t.val[count] << "]}";
                    continue;
                }
                exfile << tr_secondary_color_to_main_color.t.val[count] << ", ";
            }
            exfile.close();
        }
    }
    
    ofstream myfile("intrinsic1.json");
    if (myfile.is_open())
    {
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
    if (myfile2.is_open())
    {
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