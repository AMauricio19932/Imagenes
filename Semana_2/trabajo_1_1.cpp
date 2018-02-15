//g++ -ggdb trabajo_1_1.cpp -o out `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int scale = 2;
int delta = 0;
int ddepth = CV_16S;
RNG rng(12345);

float euclideanDist(Point p, Point q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int main(){

  VideoCapture cap(0);
  namedWindow( "Frame", CV_WINDOW_AUTOSIZE );
  namedWindow( "gray_image", CV_WINDOW_AUTOSIZE );
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
  Mat frame;
  //cap >> frame;
  bool flag = 1;

  VideoWriter outputVideo;
  //cap >> frame;
  //outputVideo.open(NAME, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
  int frame_width=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);

    if (!video.isOpened())
    {
        cout  << "Could not open the output video for write: " << endl;
        return -1;
    }

  while(1){

    int64 start = cv::getTickCount();

    if (flag) {
      cap >> frame;
      if (frame.empty())
        break;

      int ite = (int)(gray_image1.rows/180);

      // Display the resulting frame
      cvtColor( frame, gray_image2, COLOR_BGR2GRAY );

      filter2D(gray_image2, gray_image1, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
      //filter2D(gaus, gaus, -1 , kernel2, anchor, delta, BORDER_DEFAULT );
      filter2D(gray_image1, gray_image1, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
      cv::Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), anchor);

      //std::cout << ite << '\n';
      dilate( gray_image1, gray_image1, element1, Point(-1, -1), ite , 1, 1);
      adaptiveThreshold(gray_image1, gray_image1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 2);
      //dilate( gray_image1, gray_image1, element, Point(-1, -1), ite, 1, 1);
      cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), anchor);
      dilate( gray_image1, gray_image1, element2, Point(-1, -1), ite*2, 1, 1);

      //dilate( gray_image1, gray_image1, element1, Point(-1, -1), ite, 1, 1);
      imshow( "Frame", gray_image1 );

      filter2D(gray_image2, gray_image2, -1 , kernel2, anchor, delta, BORDER_DEFAULT );
      Canny( gray_image2, gray_image2, 30, 30*3, 3 );
      imshow( "Frame 3", gray_image2 );
      gray_image2.copyTo(gray_image1, gray_image1);
      imshow( "Frame 4", gray_image1 );

      //SimpleBlobDetector detector(params); // = cv::SimpleBlobDetector::create();
      vector<KeyPoint> keypoints;
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;

      //detector.detect(gray_image1, keypoints);
      findContours(gray_image1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
      vector<RotatedRect> minEllipse( contours.size() );
      vector<RotatedRect> minEllipseFinal( contours.size() );

       for( int i = 0; i < contours.size(); i++ )
       { //minRect[i] = minAreaRect( Mat(contours[i]) );
            if( contours[i].size() > 5 )
            {
              minEllipse[i] = fitEllipse( Mat(contours[i]) );
            }
            //std::cout << minEllipse[i].center << '\n';

       }
      vector<int> hist( contours.size() );
      for( int i = 0; i < contours.size(); i++ ) {
        hist[i] = 0;
        for( int j = 0; j < contours.size(); j++ ) {
          if (euclideanDist(minEllipse[i].center, minEllipse[j].center) < 3)  hist[i]++;
        }
      }

      //std::cout << "--------------------------" << '\n';
      for( int i = 0; i< contours.size(); i++ )
      {
        //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        if (hist[i] > 2) {
          ellipse( frame, minEllipse[i], Scalar(255, 0, 255), 2, 8 );
        }

      }
      /*for (size_t i = 0; i < keypoints.size(); ++i){
        circle(frame, keypoints[i].pt, 4, Scalar(255, 0, 255), -1);
        //std::cout<<"[" << keypoints[i].pt.x << " , " << keypoints[i].pt.y << ']';
      }*/
      //std::cout << '\n';
      double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
      //std::cout << "FPS : " << fps << std::endl;
      std::ostringstream sstream;
      sstream << fps;
      putText(frame, sstream.str(), Point(20, 40), 2, 1.5, Scalar(255,0,0), 2,0);
      imshow( "gray_image", frame );
      video.write(frame);

      //flag = 0;
    }

    char c = (char)waitKey(25);

    if(c == 27)
      break;

    if(c == 'p'){

      flag = 1 - flag;
    }

  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}
