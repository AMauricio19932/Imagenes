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
#include "iterativo.cpp"

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat frame;
int nuf = 45;
vector<Point2f> arrange, int_arrange, org_arrange;
vector<vector<Point2f> > int_points, fin_points;
vector<Mat> int_frames, fin_frames;

// Calibration
Mat cameraMatrix, distCoeffs;
Mat rview, output;
Mat lambda( 3, 3, CV_32FC1 );

int main(){
  vector<Point2f> corners;
  corners.push_back(Point2f(1000, 1000)); //min
  corners.push_back(Point2f(0, 0)); //max
  double  error, AcErr = 0;;
  bool first_flag = 1, pause = 0;
  double tda = 0;
  int nframes = 0;

  VideoCapture cap("calibrationVideos/lfc_ring_cut.avi");
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;}

  cap >> frame;

  Size patternsize(5, 4);
  Size imageSize = frame.size();
  Mat bg_density(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
  Mat bg_normalizado(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));

  while(1){
    char c = (char)waitKey(25);
    if (!pause){
      nframes++;
      int64 start = cv::getTickCount();
      cap >> frame;
      if (frame.empty())  break;
      detect_rings(frame, int_points, first_flag, corners, int_frames);
      write_time(frame, start, tda, nframes);
      imshow( "output", frame );
    }
    if(c == 27)  break;
    if(c == 'p') pause = 1 - pause;
  }

  cap.release();
  destroyAllWindows();

  see_density(bg_density, int_points, "No_normalizado");
  normalize_density_nuf(corners, int_points, fin_points, int_frames, fin_frames, nuf);
  //get_random_samples(int_points, fin_points, int_frames, fin_frames, nuf);
  see_density(bg_normalizado, fin_points, "Normalizado");

  std::cout << "size " << fin_frames.size() <<" "<< fin_points.size() <<'\n';

  double rms = calibrate_function(patternsize, imageSize, 44.3, cameraMatrix, distCoeffs, fin_points);
  fin_points.clear();

  // iterativo
  for (size_t i = 0; i < 5; i++) {

    for (size_t f = 0; f < fin_frames.size(); f++) {
      undistort(fin_frames[f], rview, cameraMatrix, distCoeffs);
      imshow( "output", fin_frames[f] );
      std::cout << "aca 1" << '\n';
      vector<Point2f> points = core_get_keypoints(rview, 200, 1, 1);
      first_function(points, arrange);

      frontImageRings(lambda, rview, output, arrange, patternsize);
      imshow( "output2", output );
      std::cout << "aca 2 -> " << points.size()<< '\n';
      vector<Point2f> points2 = core_get_keypoints(output, 700, 1, 2);

      std::cout << "aca 3 -> " << points2.size()<< '\n';
      while (1) {
        /* code */
      }
      first_function(points2, arrange);
      std::cout << "aca 4" << '\n';
      int_arrange = RectCorners(arrange, patternsize);
      distortPoints(arrange, cameraMatrix, distCoeffs, lambda, patternsize);
      distortPoints(int_arrange, cameraMatrix, distCoeffs, lambda, patternsize);
      vector<Point2f> points3 = get_keypoints(fin_frames[f]);
      first_function(points3, org_arrange);
      for (size_t i = 0; i < arrange.size(); i++){
        arrange[i].x = (arrange[i].x + int_arrange[i].x + org_arrange[i].x)/3;
        arrange[i].y = (arrange[i].y + int_arrange[i].y + org_arrange[i].y)/3;
      }
      std::cout << "aca 5" << '\n';
      fin_points.push_back(arrange);
    }
    double rms = calibrate_function(patternsize, imageSize, 44.3, cameraMatrix, distCoeffs, fin_points);
  }

  return 0;
}
