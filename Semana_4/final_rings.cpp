// g++ -ggdb final_rings.cpp -o out `pkg-config --cflags --libs opencv`

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "rings_functions.cpp"

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat frame;
int nuf = 100;
vector<Point2f> arrange;
vector<vector<Point2f> > int_points, fin_points;

int main(){
  vector<Point2f> corners;
  corners.push_back(Point2f(1000, 1000)); //min
  corners.push_back(Point2f(0, 0)); //max
  bool first_flag = 1, pause = 0;
  double tda = 0;
  int nframes = 0;

  VideoCapture cap("calibrationVideos/lfc_ring_cut.avi");
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;}

  cap >> frame;

  Mat bg_density(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
  Mat bg_normalizado(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));

  while(1){
    char c = (char)waitKey(25);
    if (!pause){
      nframes++;
      int64 start = cv::getTickCount();
      cap >> frame;
      if (frame.empty())  break;
      vector<Point2f> points = get_keypoints(frame);

      if (points.size() == 20){
        if(first_flag){
          first_flag = 0;
          arrange.clear();
          first_function(frame, points, arrange);
          if(arrange.size() != 20){
            first_flag = 1;
          }else{
            limits_density(int_points, arrange, corners);
            trace_line(frame, arrange);
            //update_density(bg_density, arrange, "Real Time");
          }
        }else{
          matching_normal(arrange, points);
          if(get_opposite(points, points[0]) == 19 && arrange.size() == 20){
            limits_density(int_points, arrange, corners);
            trace_line(frame, arrange);
            //update_density(bg_density, arrange, "Real Time");
          }else{
            first_flag = 1;
          }
        }
      } else {
        first_flag = 1;
      }
      write_time(frame, start, tda, nframes);
      imshow( "output", frame );
    }

    if(c == 27)  break;
    if(c == 'p') pause = 1 - pause;
  }

  see_density(bg_density, int_points, "No_normalizado");
  normalize_density_nuf(corners, int_points, fin_points, nuf);
  see_density(bg_normalizado, fin_points, "Normalizado");

  cap.release();
  destroyAllWindows();
  return 0;
}
