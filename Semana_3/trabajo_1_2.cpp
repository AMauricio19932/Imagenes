//g++ -ggdb trabajo_1_2.cpp -o out `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

int scale = 2;
int delta = 0;
int ddepth = CV_16S;
stringstream ss;
RNG rng(12345);
Point2f axis_x, axis_y;
vector<Point2f> frs(20);
float dis_x, dis_y, max_dis;
bool first_flag = 1, was_lose = 0;

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
  params.minArea = 300;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.7;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.9;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;

  SimpleBlobDetector detector(params);//para opencv 2
  //Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);//para opencv 3

  //VideoCapture cap(0);
  VideoCapture cap("calibrationVideos/LifeCam_rings.avi");
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

  int number_of_frame = 0;
  vector<int> calibration_frames;

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
      detector.detect(gray_image1, keypoints);//para opencv 2
      //detector->detect(gray_image1, keypoints); //para opencv 3

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

      bool reboot = 0;
      if (pointCentre.size() == 20){
        int op_ind;
        int indx;

        if(was_lose){
          was_lose = 0;
          float min_dis = 0;
          vector<Point2f> back_up = pointCentre;

          for (size_t i = 0; i < frs.size(); i++) {
            max_dis = 1000;
            for (size_t j = 0; j < back_up.size(); j++) {
              if(max_dis > euclideanDist(back_up[j], frs[i])){
                max_dis = euclideanDist(back_up[j], frs[i]);
                indx = j;
              }
            }
            if (dis_x < max_dis){
              std::cout << dis_x*2 << " " << max_dis << " " << i <<" "<< indx <<'\n';
              reboot = 1;
              break;
            }
            frs[i] = back_up[indx];
            back_up.erase (back_up.begin()+indx);
          }

          for (size_t j = 1; j < frs.size(); j++) {
            if(min_dis < euclideanDist(frs[0], frs[j])){
              min_dis = euclideanDist(frs[0], frs[j]);
              op_ind = j;
            }
          }

          cout << number_of_frame <<" "<< reboot <<" "<< op_ind <<endl;
          if(!reboot && op_ind == 19){
            first_flag = 0; // no se rebootea
          }
        }

        if(first_flag){
          first_flag = 0;
          frs.clear();
          dis_x = euclideanDist(pointCentre[1], pointCentre[0]);
          axis_x.x = (pointCentre[1].x - pointCentre[0].x)/dis_x;
          axis_x.y = (pointCentre[1].y - pointCentre[0].y)/dis_x;

          max_dis = 0;
          for (size_t j = 2; j < pointCentre.size(); j++) {
            if(max_dis < euclideanDist(pointCentre[0], pointCentre[j])){
              max_dis = euclideanDist(pointCentre[0], pointCentre[j]);
              op_ind = j;
            }
          }

          //std::cout << "DIRECCION " << axis_x.x <<" "<< axis_x.y  << '\n';

          int b_x = 0;
          bool flag_end = 0, next_row;

          frs.push_back(pointCentre[0]);
          frs.push_back(pointCentre[1]);

          // calculo de la primera linea
          k = 1;
          while (!flag_end){
            for (size_t j = 2; j < pointCentre.size(); j++){
              flag_end = 1;
              if (euclideanDist(pointCentre[k] + dis_x*axis_x, pointCentre[j]) < dis_x/4.0){
                flag_end = 0;
                dis_x = euclideanDist(pointCentre[j], pointCentre[k]);
                k = j;
                frs.push_back(pointCentre[k]);
                break;
              }
            }
          }

          dis_y = euclideanDist(pointCentre[op_ind], pointCentre[k]);
          axis_y.x = (pointCentre[op_ind].x - pointCentre[k].x)/dis_y;
          axis_y.y = (pointCentre[op_ind].y - pointCentre[k].y)/dis_y;
          dis_y = dis_x;

          //CORRECCION DE EJES
          if (abs(axis_x.x) > abs(axis_x.y)){ // si es eje x
            if(axis_x.x < 0){// invertido
              axis_x.x = -axis_x.x;
              axis_x.y = -axis_x.y;
              reverse(frs.begin(),frs.end());
              b_x = k; // siguiente eje
            }
          }

          flag_end = 0;
          next_row = 1;

          while(!flag_end){
            //std::cout << "elemento inicial "<< b_x << '\n';
            if (next_row){
              next_row = 0;
              for (size_t j = 2; j < pointCentre.size(); j++) {
                flag_end = 1;
                if (euclideanDist(pointCentre[b_x] + dis_y*axis_y, pointCentre[j]) < dis_y/4.0){
                  dis_y = euclideanDist(pointCentre[b_x], pointCentre[j]);
                  flag_end = 0;
                  b_x = j;
                  k = b_x;
                  //std::cout << "elemento y "<< k << '\n';
                  frs.push_back(pointCentre[b_x]);
                  break;
                }
              }
            }

            for (size_t j = 2; j < pointCentre.size(); j++) { //eje x
              next_row = 1;
              if (euclideanDist(pointCentre[k] + dis_x*axis_x, pointCentre[j]) < dis_x/4.0){
                next_row = 0;
                k = j;
                //std::cout << "elemento x "<< k << '\n';
                frs.push_back(pointCentre[k]);
                break;}
            }

          }
        }

        max_dis = 0;
        for (size_t j = 1; j < frs.size(); j++) {
          if(max_dis < euclideanDist(frs[0], frs[j])){
            max_dis = euclideanDist(frs[0], frs[j]);
            op_ind = j;
          }
        }

        if(frs.size() == 20 && op_ind == 19){//frs[0].x < frs.back().x
          // TRACKING
          float min_frs;
          float new_d_frs;

          for (size_t i = 0; i < frs.size(); i++) {
            min_frs = 1000;

            for (size_t j = 0; j < pointCentre.size(); j++) {
              new_d_frs = euclideanDist(frs[i], pointCentre[j]);
              if(min_frs > new_d_frs){
                min_frs = new_d_frs;
                indx = j;//nuevo punto
              }
            }
            frs[i] = pointCentre[indx];
            pointCentre.erase (pointCentre.begin()+indx);
          }

          for (size_t i = 0; i < frs.size(); i++){
            ss << i;
            putText(frame, ss.str(), frs[i], 1.7, 2.0, Scalar(0,0,0), 0.1, 0);
            ss.str("");}

          calibration_frames.push_back(number_of_frame);
        }else{
          if(frs.size() ==  20)
            was_lose = 1;
          first_flag = 1;
        }
      }else{
        if(frs.size() == 20)
          was_lose = 1;
        first_flag = 1;
      }

      number_of_frame +=1;
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

    if(c == 'p')
      flag = 1 - flag;

  }
  for (size_t i = 0; i < calibration_frames.size(); i++) {
    std::cout << calibration_frames[i] << " ";
  }
  std::cout << '\n' << calibration_frames.size() << '\n';
  cap.release();
  destroyAllWindows();

  return 0;
}
