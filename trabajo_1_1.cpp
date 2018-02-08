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

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("calibration_mslifecam.avi");
  namedWindow( "Frame", CV_WINDOW_AUTOSIZE );
  namedWindow( "gray_image", CV_WINDOW_AUTOSIZE );
  Mat kernel1 = (Mat_<double>(5,5) <<
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1) / 25;
  Mat kernel3 = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
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
  while(1){

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // Display the resulting frame
    filter2D(frame, gaus, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
    filter2D(gaus, gaus, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
    filter2D(gaus, gaus, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
    cvtColor( gaus, gray_image1, COLOR_BGR2GRAY );
    //Mat bgr[3];
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), anchor);
    //filter2D(gray_image2, gray_image1, -1 , str_el, anchor, delta, BORDER_DEFAULT );
    int ite = (int)(gray_image1.rows/180);
    //std::cout << ite << '\n';
    dilate( gray_image1, gray_image1, element, Point(-1, -1), ite , 1, 1);
    adaptiveThreshold(gray_image1, gray_image1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 20);
    dilate( gray_image1, gray_image1, element, Point(-1, -1), ite, 1, 1);
    imshow( "Frame", gray_image1 );


    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(gray_image1, keypoints);

    int j, i;
    for ( i = 1; i < keypoints.size(); ++i){
      j = i - 1;
      line(frame, keypoints[j].pt, keypoints[i].pt, Scalar(0, 255, 0), 3);
    }
    for ( i = 0; i < keypoints.size(); ++i){
      circle(frame, keypoints[i].pt, 4, Scalar(255, 0, 255), -1);
    }
    imshow( "gray_image", frame );


    char c = (char)waitKey(25);
    if(c == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}
