//g++ -ggdb facedetect.cpp -o facedetect `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "rings_functions.cpp"
#include "iterativo.cpp"
using namespace std;
using namespace cv;

int main(){
  bool flag_tipo = 0;
  double  error, AcErr = 0;;

  //VideoCapture cap("lfc_chess.avi");
  //VideoCapture cap("ps3_chess.avi");
  //VideoCapture cap("lfc_assm.avi");
  VideoCapture cap("ps3_assm.avi");

  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  int frame_width=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  //VideoWriter video("outCali.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
  Mat frame, gray;

  cap >> frame;
  bool patternfound;
  vector<Point2f> corners, corners2, corners3;
  vector<vector<Point2f> > imagePoints;
  Size imageSize = frame.size();
  std::vector<int> indImage;
  bool flag = 1, flag2 = 0;
  Size patternsize;
  int con = 0;
  if(flag_tipo)
    patternsize = Size(7,5);
  else
    patternsize = Size(4,11);

  int it = 0, numcap = 1;
  while(1){
    if (flag) {
      cap >> frame;
      con++;
      if (frame.empty()) break;
      cvtColor( frame, gray, COLOR_BGR2GRAY );
      corners.clear();
      if(flag_tipo) patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
      else patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );
      //video.write(frame);
      if(patternfound){
        if(flag_tipo) cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
        drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);
        error = colinialidad (corners, patternsize);
      }
      imshow( "frame", frame );
      //video.write(frame);
      numcap++;
    }

    char c = (char)waitKey(25);
    if(c == 27 || imagePoints.size() == 25) break;
    else if(c == 'p')flag = 1 - flag;
    else if (c == 'c' && patternfound) {
      imagePoints.push_back(corners);
      it++;
      indImage.push_back(con);
      std::cout << "cap #"<< it << '\n';
      error = sqrt(error) / ((patternsize.width)*patternsize.height) ;
      AcErr += error;
      error = AcErr/it;
      std::cout << "error de colinealidad = "<< error << '\n';
      //video.write(frame);
      flag2 = 1;
      patternfound = 0;
    }
  }
  Mat bg_density(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
  see_density(bg_density, imagePoints, "No_normalizado");

  for (size_t i = 0; i < indImage.size(); i++) {
    std::cout << indImage[i] << ", ";
  }

  cap.release();
  vector<Mat> rvecs, tvecs;
  Mat cameraMatrix, distCoeffs;
  vector<Point3f> vectorPoints;
  vector<vector<Point3f> > objectPoints;
  Mat view, rview, output;
  Mat lambda( 3, 3, CV_32FC1 );

  if(flag_tipo)
    for( int i = 0; i < patternsize.height; ++i )
      for( int j = 0; j < patternsize.width; ++j )
        vectorPoints.push_back(Point3f(float(j)*27, float(i)*27, 0));
  else
    for( int i = 0; i < patternsize.height; i++ )
        for( int j = 0; j < patternsize.width; j++ )
            vectorPoints.push_back(Point3f(float((2*j + i % 2))*35, float(i)*35, 0));
  for (int k = 0; k < imagePoints.size(); k++)
    objectPoints.push_back(vectorPoints);

  cameraMatrix = Mat::eye(3, 3, CV_64F);
  distCoeffs = Mat::zeros(8, 1, CV_64F);
  double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
  std::cout << "error = "<< rms << '\n';
  std::cout << "cameraMatrix = "<< cameraMatrix << '\n';
  std::cout << "distCoeffs = "<< distCoeffs << '\n';

  for (size_t cali = 0; cali < 5 && flag2; cali++) {
    flag = 1;
    //VideoCapture capTest("lfc_chess.avi");
    //VideoCapture capTest("ps3_chess.avi");
    //VideoCapture capTest("lfc_assm.avi");
    VideoCapture capTest("ps3_assm.avi");

    AcErr = 0;
    int canE = 0;
    numcap = 0;
    it = 0;
    con = 0;
    imagePoints.clear();
    while(1){
      if (flag) {
        capTest >> view;
        con++;
        if(view.empty()) break;
        undistort(view, rview,cameraMatrix, distCoeffs);
        cvtColor( rview, gray, COLOR_BGR2GRAY );
        corners.clear();
        if(flag_tipo)
          patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        else
          patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );
        //video.write(frame);
        if(patternfound){
          if(flag_tipo) cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

          if(flag_tipo) frontImageChess(lambda, rview, output, corners, patternsize);
          else frontImageAsime(lambda, rview, output, corners, patternsize);

          cvtColor( output, gray, COLOR_BGR2GRAY );
          error = colinialidad (corners, patternsize);
          corners.clear();
          if(flag_tipo) patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
          else patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );

          if (patternfound) {
            if(flag_tipo) cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

            corners3.clear();
            if(flag_tipo) corners3 = RectCorners(corners, patternsize);
            else corners3 = RectCornersAsime (corners, patternsize);
            //std::cout << corners3.size() << '\n';
            //std::cout << patternsize << '\n';
            drawChessboardCorners(output, patternsize, Mat(corners3), patternfound);
            //std::cout << "error 1" << '\n';
            imshow("Image patron", output);
            //std::cout << "error 2" << '\n';

            distortPoints(corners, cameraMatrix, distCoeffs, lambda, patternsize);
            //std::cout << "error 3" << '\n';
            distortPoints(corners3, cameraMatrix, distCoeffs, lambda, patternsize);

            cvtColor( view, gray, COLOR_BGR2GRAY );
            corners2.clear();
            if(flag_tipo) patternfound = findChessboardCorners(gray, patternsize, corners2, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
            else patternfound = findCirclesGrid( gray, patternsize, corners2, CALIB_CB_ASYMMETRIC_GRID );
            if (patternfound) {
              if(flag_tipo) cornerSubPix( gray, corners2, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
              for (size_t i = 0; i < corners2.size(); i++) {
                corners[i] = (corners[i] + corners2[i] + corners3[i])/3;
                //corners[i] = (corners[i] + corners2[i])/2;
              }
            }
            //std::cout << "error 4" << '\n';
            drawChessboardCorners(view, patternsize, Mat(corners), patternfound);
            numcap ++;
            canE ++;
          }
          imshow("Image Normal", view);
          imshow("Image Modif", rview);
        }
      }
      char c = (char)waitKey(25);
      if( c  == 27 || imagePoints.size() == 25)
          break;
      else if(c == 'p'){
        flag = 1 - flag;
      }
      else if (indImage[it] == con && patternfound) {
        imagePoints.push_back(corners);
        it++;
        std::cout << "capr #"<< it << '\n';
        error = sqrt(error) / ((patternsize.width)*patternsize.height) ;
        AcErr += error;
        error = AcErr/it;
        std::cout << "error de colinealidad = "<< error << '\n';
        //video.write(frame);
        flag2 = 1;
        patternfound = 0;
      }
    }
    
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    distCoeffs = Mat::zeros(8, 1, CV_64F);
    objectPoints.clear();
    for (int k = 0; k < imagePoints.size(); k++) objectPoints.push_back(vectorPoints);

    rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "error = "<< rms << '\n';
    std::cout << "cameraMatrix = "<< cameraMatrix << '\n';
    std::cout << "distCoeffs = "<< distCoeffs << '\n';
    capTest.release();
  }
  destroyAllWindows();
  return 0;
}
