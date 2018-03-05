//g++ -ggdb facedetect.cpp -o facedetect `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int scale = 2;
int delta = 0;
int ddepth = CV_16S;

int main(){
  bool flag_tipo = 1;
  //VideoCapture cap("videos calibracion/LifeCam_asymmetric_circles.avi");
  VideoCapture cap("videos calibracion/PS3_chess.avi");
  //VideoCapture cap("videos calibracion/LifeCam_chess.avi");

  //VideoCapture capTest("calibrationVideos/calibration_test_asymmetric_2.avi");
  //VideoCapture cap("calibrationVideos/calibration_test_chessboard_1.avi");
  //VideoCapture capTest("calibrationVideos/calibration_test_chessboard_2.avi");
  //VideoCapture cap(0);
  //VideoCapture cap("outCali_02.avi");
  //namedWindow( "Frame", CV_WINDOW_AUTOSIZE );
  //namedWindow( "gray_image", CV_WINDOW_AUTOSIZE );
  Mat kernel1 = (Mat_<double>(5,5) <<
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1) / 25;
  //Mat kernel3 = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat kernel2 = (Mat_<double>(5,5) <<
  2, 4, 5, 4, 2,
  4, 9, 12, 9, 4,
  5, 12, 15, 12, 5,
  4, 9, 12, 9, 4,
  2, 4, 5, 4, 2);

  kernel2 = kernel2/159.0;

  Point anchor;
  double delta;
  //int ddepth;
  Mat gray_image1, gray_image2, gaus;

  anchor = Point( -1, -1 );
  delta = 0;
  //ddepth = -1;
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  int frame_width=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  VideoWriter video("outCali.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
  Mat frame, gray, frame2;

  cap >> frame;
  bool patternfound;
  vector<Point2f> corners;
  vector<vector<Point2f> > imagePoints;
  Size imageSize = frame.size();
  bool flag = 1, flag2 = 0;
  Size patternsize;
  if(flag_tipo)
    patternsize = Size(9,6);
  else
    patternsize = Size(4,11);

  int it = 0, numcap = 1;
  while(1){
    if (flag) {
      cap >> frame;
      if (frame.empty()) break;
      cvtColor( frame, gray, COLOR_BGR2GRAY );
      corners.clear();
      if(flag_tipo)
        patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
      else
        patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );
      video.write(frame);
      if(patternfound){
        if(flag_tipo)
          cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
        drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);
      }
      imshow( "gray_image", frame );
      video.write(frame);
      numcap++;

    }

    char c = (char)waitKey(25);

    if(c == 27 || imagePoints.size() == 55) break;

    else if(c == 'p'){
      flag = 1 - flag;
    } else if (numcap % 60 == 0 && patternfound) {
      imagePoints.push_back(corners);
      it++;
      std::cout << "cap #"<< it << '\n';
      //video.write(frame);
      flag2 = 1;
      patternfound = 0;
    }

  }

  std::cout << "size vector = "<< imagePoints.size() << '\n';
  //std::cout << "size height = "<< patternsize.height << ", size width = "<< patternsize.width <<'\n';
  vector<Point3f> vectorPoints;
  vector<vector<Point3f> > objectPoints;
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix, distCoeffs;
  cameraMatrix = Mat::eye(3, 3, CV_64F);

  distCoeffs = Mat::zeros(8, 1, CV_64F);
  if(flag2){
    if(flag_tipo)
      for( int i = 0; i < patternsize.height; ++i )
        for( int j = 0; j < patternsize.width; ++j )
          vectorPoints.push_back(Point3f(float( j), float( i ), 0));
    else
      for( int i = 0; i < patternsize.height; i++ )
          for( int j = 0; j < patternsize.width; j++ )
              vectorPoints.push_back(Point3f(float((2*j + i % 2)), float(i), 0));

    for (int k = 0; k < imagePoints.size(); k++)
      objectPoints.push_back(vectorPoints);

    //td::cout << "size vector = "<< objectPoints.size() << '\n';
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "error = "<< rms << '\n';
    std::cout << "cameraMatrix = "<< cameraMatrix << '\n';
    std::cout << "distCoeffs = "<< distCoeffs << '\n';

    Mat view, rview, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
    imageSize, CV_16SC2, map1, map2);
    flag = 1;
    std::cout << "/* fin calibracion */" << '\n';
    while(1)
    {
      if (flag) {
        cap >> view;
        if(view.empty()) break;
        remap(view, rview, map1, map2, INTER_LINEAR);
        imshow("Image Normal", view);
        imshow("Image Modif", rview);
        video.write(rview);
      }
        char c = (char)waitKey(25);
        if( c  == 27 || c == 'q' || c == 'Q' )
            break;
        else if(c == 'p'){
          flag = 1 - flag;
        }
    }
  }
  cap.release();
  // Closes all the frames
  destroyAllWindows();

  return 0;
}
