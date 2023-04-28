# Kinect-calib-and-capture
This program calibrates 2 Azure Kinect cameras, saves their intrinsic camera matrices, the extrinsic camera matrices, and the depth and color images.<br />
## Dependencies
The following (non-standard) libraries are required:<br />
OpenCV (only tested with version 4.7.0) <br />
k4a (only tested with 1.4.1, also known as the Microsoft Azure Kinect sensor SDK, can be installed using NuGet)
## How to run
After compiling the executable has to be run with one argument, a `0` or a `1` <br />
If the argument `1` is passed the program will run with stereo calibration and generate a new extrinsic.json <br />
If the argument `0` is passed the program will run using the previously generated stereo calibration data (extrinsic.json) and thus skip the stereo calibration process <br /> <br />
Upon running the executable the captures that will be used for the point cloud(s) are immediately made <br />
Subsequently the program pauses, and explains the calibration process, before taking a capture for the calibration the user has to give input <br />
After 50 captures the calibration is complete <br />
The chessboard used is from: [link](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/tree/develop/examples/green_screen) <br /> <br />
**NOTE: the value of the CHESSBOARD_SQUARE_SIZE define at the top of the testing-kinect.cpp file has to match the square size (in mm) of the chessboard being used for calibration**
## Output
The program generates the following output: <br /> <br />
color1.jpg&nbsp;&nbsp;&nbsp;&nbsp; *the color image of the master camera* <br /> <br />
color2.jpg&nbsp;&nbsp;&nbsp;&nbsp; *the color image of the subordinate camera* <br /> <br />
depth1.png&nbsp;&nbsp;&nbsp;&nbsp; *the depth image of the master camera* <br /> <br />
depth2.png&nbsp;&nbsp;&nbsp;&nbsp; *the depth image of the subordinate camera* <br /> <br />
intrinsic1.json&nbsp;&nbsp;&nbsp;&nbsp; *the camera intrinsic matrix of the master camera* <br /> <br />
intrinsic2.json&nbsp;&nbsp;&nbsp;&nbsp; *the camera intrinsic matrix of the subordinate camera* <br /> <br />
extrinsic.json&nbsp;&nbsp;&nbsp;&nbsp; *the camera extrinsic matrices, for the transformation fom sub camera to master camera*
