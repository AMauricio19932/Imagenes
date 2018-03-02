//g++ -ggdb trabajo_1_2.cpp -o out `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

int scale = 2;
int delta = 0;
int ddepth = CV_16S;
stringstream ss;
RNG rng(12345);
Point2f axis_x, axis_y;
vector<Point2f> frs;
float dis = 0.0;

float euclideanDist(Point p, Point q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int main(){
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;

  // Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 200;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 150;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.7;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.9;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;

  //SimpleBlobDetector detector(params);//para opencv 2
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);//para opencv 3

  //VideoCapture cap(0);
  //VideoCapture cap("videos calibracion/2018-03-01-213635.webm");
  //VideoCapture capTest("videos calibracion/2018-03-01-213424.webm");
  VideoCapture capTest("videos calibracion/ChessBoard.wmv");
  VideoCapture cap("videos calibracion/Rings.wmv");
  //VideoCapture cap("calibration_mslifecam.avi");
  //VideoCapture cap("realsense_RGB.avi");
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
  Mat frame;
  //cap >> frame;
  bool flag = 1;
  bool first_flag = 1;

  VideoWriter outputVideo;
  //cap >> frame;
  //outputVideo.open(NAME, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
  int frame_width=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), anchor);
  cv::Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), anchor);

  VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
  int ite = (int)(frame_height/180);

  if (!video.isOpened()){
      cout  << "Could not open the output video for write: " << endl;
      return -1;}

  double tda = 0;
  int con = 0, k, p;
  char c;
  float h = 0, w = 0, h2, w2, d, dm;
  int it = 0;
  for (size_t i = 0; i < 500; i++) {
    cap >> frame;
  }
  vector<Point2f> corners; //this will be filled by the detected corners
  vector<vector<Point2f> > imagePoints;
  bool flag2 = 0;
  Size imageSize = frame.size();
  Size patternsize(5,4);
  while(1){
    int64 start = cv::getTickCount();
    c = (char)waitKey(25);

    if (flag) {
      cap >> frame;
      con++;
      if (frame.empty())
        break;

      cvtColor( frame, gray_image2, COLOR_BGR2GRAY );

      filter2D(gray_image2, gray_image1, -1 , kernel1, anchor, delta, BORDER_DEFAULT );
      filter2D(gray_image1, gray_image1, -1 , kernel1, anchor, delta, BORDER_DEFAULT );

      adaptiveThreshold(gray_image1, gray_image1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 2);

      erode( gray_image1, gray_image1, element1, Point(-1, -1), 1, 1, 1);

      dilate( gray_image1, gray_image1, element1, Point(-1, -1), 1, 1, 1);

      vector<KeyPoint> keypoints;
      //detector.detect(gray_image1, keypoints);//para opencv 2
      detector->detect(gray_image1, keypoints); //para opencv 3

      h = 0, w = 0;
      for (size_t i = 0; i < keypoints.size(); ++i){
        h += keypoints[i].pt.x;
        w += keypoints[i].pt.y;
      }
      h /=  keypoints.size();
      w /=  keypoints.size();
      circle(frame, Point2f(h,w), 4, Scalar(255, 0, 255), -1);

      if(keypoints.size() > 20){
        k = keypoints.size() - 20;
        dm = 0;
        for (int i = 0; i < k; i++) {
          for( int j = 0; j < keypoints.size(); j++ ) {
            d = euclideanDist(Point2f(h,w), keypoints[j].pt);
            if (d > dm) {
              dm = d;
              p = j;
            }
          }
          keypoints.erase(keypoints.begin() + p);
        }
      }

      vector<Point2f> pointCentre( keypoints.size() );
      float xm = frame_width, xM = 0, ym = frame_height, yM = 0;
      for (size_t i = 0; i < keypoints.size(); ++i){
        if (xm > keypoints[i].pt.x) xm = keypoints[i].pt.x;
        if (xM < keypoints[i].pt.x) xM = keypoints[i].pt.x;
        if (ym > keypoints[i].pt.y) ym = keypoints[i].pt.y;
        if (yM < keypoints[i].pt.y) yM = keypoints[i].pt.y;
        //circle(frame, keypoints[i].pt, 2, Scalar(0, 255, 0), -1);
        pointCentre[i] = Point2f(0,0);
      }
      xM += 3; xm -= 3;
      yM += 3; ym -= 3;


      Canny( gray_image1, gray_image2, 50, 50 * 3, 3 );

      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;

      findContours(gray_image2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
      vector<RotatedRect> minEllipse( contours.size() );
      vector<RotatedRect> minEllipseFinal( contours.size() );
      RotatedRect cir;
      k = 0;
      h = 0, w = 0, h2, w2;

       for( int i = 0; i < contours.size(); i++ )  {
         if( contours[i].size() > 5 ){
           cir = fitEllipse( Mat(contours[i]) );
           if ( cir.size.height > 2 && cir.size.width > 2
             && cir.center.x > xm && cir.center.x < xM
             && cir.center.y > ym && cir.center.y < yM){
               for( int j = 0; j < keypoints.size(); j++ ) {
                 if (euclideanDist(cir.center, keypoints[j].pt) < 3) {
                   minEllipse[k] = cir;
                   h += cir.size.height;
                   w += cir.size.width;
                   k++;
                 }
                }
            }
          }

       }
       h /= k; w /= k;
       h2 = h*1.9; w2 = w*1.9;
       h = h*0.4; w = w*0.4;

      //std::cout << k << '\n';
      vector<int> hist( keypoints.size() );

      for( int i = 0; i < k; i++ ) {
        if (minEllipse[i].size.height > h && minEllipse[i].size.width > w
            && minEllipse[i].size.height < h2 && minEllipse[i].size.width < w2) {
          for( int j = 0; j < keypoints.size(); j++ ) {
            if (euclideanDist(minEllipse[i].center, keypoints[j].pt) < 3){
              hist[j]++;
              pointCentre[j] += minEllipse[i].center;
              ellipse( frame, minEllipse[i], Scalar(0, 0, 255), 2, 8 );
            }
          }
        }
      }

      //std::cout << "--------------------------" << '\n';
      //double x_min = 1000, x_max = 0, y_min = 1000, y_max = 0;
      int i = 0;
      while(i < pointCentre.size())  {
        if (hist[i] > 3) {
          pointCentre[i].x /= hist[i] ;
          pointCentre[i].y /= hist[i] ;
          circle(frame, pointCentre[i], 2, Scalar(0, 255, 0), -1);
          i++;
        } else {
          hist.erase(hist.begin() + i);
          pointCentre.erase(pointCentre.begin() + i);
        }
      }

      if (pointCentre.size() == 20){
        if(first_flag){

          first_flag = 0;
          frs.clear();
          dis = euclideanDist(pointCentre[1], pointCentre[0]);
          axis_x.x = (pointCentre[1].x - pointCentre[0].x)/dis;
          axis_x.y = (pointCentre[1].y - pointCentre[0].y)/dis;
          axis_y.x = axis_x.y;
          axis_y.y = -axis_x.x;

          frs.push_back(pointCentre[0]);
          frs.push_back(pointCentre[1]);

          int b_x = 0;
          int k = 1;
          bool flag_end = 0;
          bool flag_ny;

          while(!flag_end){
            //std::cout << "elemento inicial "<< b_x << '\n';
            for (size_t j = 2; j < pointCentre.size(); j++) { //eje x
              flag_ny = 1;
              //std::cout << "X "<< pointCentre[k]+42*axis_x <<" "<< pointCentre[j] <<" "<< euclideanDist(pointCentre[k]+42*axis_x, pointCentre[j]) << '\n';

              if (euclideanDist(pointCentre[k]+42*axis_x, pointCentre[j]) < 10){
                flag_ny = 0;
                k = j;
                //std::cout << "elemento x "<< k << '\n';
                frs.push_back(pointCentre[k]);
                break;}
            }

            if (flag_ny){ //eje y
              for (size_t j = 2; j < pointCentre.size(); j++) {
                flag_end = 1;
                //std::cout << "Y "<< pointCentre[b_x] <<" "<< pointCentre[j] <<" "<< euclideanDist(pointCentre[b_x]+42*axis_y, pointCentre[j]) << '\n';
                if (euclideanDist(pointCentre[b_x]+42*axis_y, pointCentre[j]) < 10){
                  flag_end = 0;
                  b_x = j;
                  k = b_x;
                  //std::cout << "elemento y "<< k << '\n';
                  frs.push_back(pointCentre[b_x]);
                  break;}
              }
            }
          }
        }


        vector<float> min_d;
        min_d.assign (frs.size(),1000);
        vector<float> new_d (frs.size());
        vector<int> indx (frs.size());

        for (size_t i = 0; i < frs.size(); i++) {
          for (size_t j = 0; j < pointCentre.size(); j++) {
            new_d[i] = euclideanDist(frs[i], pointCentre[j]);
            if(min_d[i] > new_d[i]){
              min_d[i] = new_d[i];
              indx[i] = j;
            }
          }
        }

        for (size_t i = 0; i < frs.size(); i++){
          ss << i;
          frs[i] = pointCentre[indx[i]];
          putText(frame, ss.str(), frs[i], 0.1, 0.5, Scalar(0,255,0), 0.1, 0);
          ss.str("");
        }
      }else{
        first_flag = 1;
      }

      double fps = (cv::getTickCount() - start) / cv::getTickFrequency() ;
      tda += fps;
      fps =  tda/con;

      //std::cout << "FPS : " << fps << std::endl;
      std::ostringstream sstream;
      sstream << fps;
      putText(frame, sstream.str(), Point(20, 40), 2, 1.5, Scalar(255,0,0), 2,0);
      imshow( "image final", frame );
      video.write(frame);

    }

    if(c == 27)
      break;

    if(c == 'p'){
      flag = 1 - flag;
    }
    else if (c == 'c') {
      imagePoints.push_back(frs);
      it++;
      std::cout << "cap #"<< it << '\n';
      flag2 = 1;
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

    for( int i = 0; i < patternsize.height; ++i )
      for( int j = 0; j < patternsize.width; ++j )
        vectorPoints.push_back(Point3f(float( j ), float( i ), 0));



    for (int k = 0; k < imagePoints.size(); k++)
      objectPoints.push_back(vectorPoints);

    //td::cout << "size vector = "<< objectPoints.size() << '\n';
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    // When everything done, release the video capture object
    std::cout << "error = "<< rms << '\n';

    Mat view, rview, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
    imageSize, CV_16SC2, map1, map2);
    flag = 1;
    while(1)
    {
      if (flag) {
        capTest >> view;
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

  destroyAllWindows();

  return 0;
}
