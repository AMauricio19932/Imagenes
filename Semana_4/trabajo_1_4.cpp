//g++ -ggdb trabajo_1_4.cpp -o tr `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int scale = 2;
int delta = 0;
int ddepth = CV_16S;
float euclideanDist(Point p, Point q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int main(){
  bool flag_tipo = 1;
  //VideoCapture cap("videos calibracion/LifeCam_asymmetric_circles.avi");

  //VideoCapture cap("videos calibracion/PS3_chess.avi");
  //VideoCapture cap("videos calibracion/PS3_asymmetric_circles.avi");
  VideoCapture cap("calibrationVideos/LifeCam_chess.avi");

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
  //VideoWriter video("outCali.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
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
      //video.write(frame);
      if(patternfound){
        if(flag_tipo)
          cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
        drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);
      }
      imshow( "gray_image", frame );
      //video.write(frame);
      numcap++;

    }

    char c = (char)waitKey(25);

    if(c == 27 || imagePoints.size() == 25) break;

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
  cap.release();
  //if(flag2){
  for (size_t cali = 0; cali < 5 && flag2; cali++) {
    std::cout << "size vector = "<< imagePoints.size() << '\n';
    //std::cout << "size height = "<< patternsize.height << ", size width = "<< patternsize.width <<'\n';
    vector<Point3f> vectorPoints;
    vector<vector<Point3f> > objectPoints;
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix, distCoeffs;
    cameraMatrix = Mat::eye(3, 3, CV_64F);

    distCoeffs = Mat::zeros(8, 1, CV_64F);
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


    Mat view, rview, map1, map2, output;
    Point2f inputQuad[4];
    Point2f outputQuad[4];
    //initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
    //getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
    //imageSize, CV_16SC2, map1, map2);
    flag = 1;
    //cap.release();
    std::cout << "/* fin calibracion */" << '\n';
    //VideoCapture capTest("videos calibracion/PS3_chess.avi");
    VideoCapture capTest("calibrationVideos/LifeCam_chess.avi");
    flag_tipo = 1;
    if(flag_tipo)
      patternsize = Size(9,6);
    else
      patternsize = Size(4,11);
    double dist[patternsize.width - 1], distP, error, AcErr = 0, xm, ym , m, b, s1, s2, s3, y, x, yD, xD, r, idistra, xa, ya, x0, y0;
    int canE = 0;
    numcap = 0;
    it = 0;
    imagePoints.clear();
    float t;
    while(1){
      if (flag) {
        capTest >> view;
        if(view.empty()) break;
        //remap(view, rview, map1, map2, INTER_LINEAR);
        undistort(view, rview,cameraMatrix, distCoeffs);
        cvtColor( rview, gray, COLOR_BGR2GRAY );
        corners.clear();
        if(flag_tipo)
          patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        else
          patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );
        //video.write(frame);
        if(patternfound){
          if(flag_tipo)
            cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
          Mat lambda( 3, 3, CV_32FC1 );

          inputQuad[0] = corners[0];
          inputQuad[1] = corners[patternsize.width - 1];
          inputQuad[2] = corners[(patternsize.height*patternsize.width) - 1];
          inputQuad[3] = corners[((patternsize.height - 1)*patternsize.width)];

          outputQuad[0] = Point2f( 80, 80 );
          outputQuad[1] = Point2f( 540 - 1,80);
          outputQuad[2] = Point2f( 540 - 1,380 - 1);
          outputQuad[3] = Point2f( 80,380 - 1);
          //lambda = Mat::zeros( rview.rows, rview.cols, rview.type() );
          lambda = getPerspectiveTransform( inputQuad, outputQuad );
          warpPerspective(rview,output,lambda,output.size() );
          //lambda = Mat::zeros( rview.rows, rview.cols, rview.type() );
          lambda = getPerspectiveTransform( outputQuad, inputQuad );
          //std::cout << "lambda = "<<lambda << '\n';

          cvtColor( output, gray, COLOR_BGR2GRAY );
          corners.clear();
          if(flag_tipo)
            patternfound = findChessboardCorners(gray, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
          else
            patternfound = findCirclesGrid( gray, patternsize, corners, CALIB_CB_ASYMMETRIC_GRID );
          if (patternfound) {
            if(flag_tipo) cornerSubPix( gray, corners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

            drawChessboardCorners(output, patternsize, Mat(corners), patternfound);
            imshow("Image patron", output);
            error = 0;
            for( int i = 0; i < patternsize.height; ++i ){
              xm = 0;
              ym = 0;
              for( int j = 0; j < patternsize.width; ++j ){
                x = lambda.at <double>(0,0) * corners[j + (i*patternsize.width)].x;
                x += lambda.at <double>(0,1) * corners[j + (i*patternsize.width)].y;
                x += lambda.at <double>(0,2);

                y = lambda.at <double>(1,0) * corners[j + (i*patternsize.width)].x;
                y += lambda.at <double>(1,1) * corners[j + (i*patternsize.width)].y;
                y += lambda.at <double>(1,2);

                t = lambda.at <double>(2,0) * corners[j + (i*patternsize.width)].x;
                t += lambda.at <double>(2,1) * corners[j + (i*patternsize.width)].y;
                t += lambda.at <double>(2,2);

                x /= t;
                y /= t;

                x0  = (x - cameraMatrix.at <double>(0,2))/cameraMatrix.at <double>(0,0);
                y0  = (y - cameraMatrix.at <double>(1,2))/cameraMatrix.at <double>(1,1);
                x = x0;
                y = y0;
                r = (x*x) + (y*y);
                idistra = (1 + (distCoeffs.at <double>(0,0)*r) + (distCoeffs.at <double>(0,1)*r*r) + (distCoeffs.at <double>(0,4)*r*r*r));
                x = x*idistra + ((2*distCoeffs.at <double>(0,2)*x*y) + (distCoeffs.at <double>(0,3)*(r + (2*x*x))));
                y = y*idistra + ((2*distCoeffs.at <double>(0,3)*x*y) + (distCoeffs.at <double>(0,2)*(r + (2*y*y))));
                /*float e = 1;
                for (size_t k = 0; k < 500 && e > 0.0; k++) {
                  r = (x*x) + (y*y);
                  idistra = 1/(1 + (distCoeffs.at <double>(0,0)*r) + (distCoeffs.at <double>(0,1)*r*r) + (distCoeffs.at <double>(0,4)*r*r*r));
                  //r *=r;
                  //xD = x/(1 + (distCoeffs.at <double>(0,0)*r) + (distCoeffs.at <double>(0,1)*r*r) + (distCoeffs.at <double>(0,4)*r*r*r));
                  //yD = y/(1 + (distCoeffs.at <double>(0,0)*r) + (distCoeffs.at <double>(0,1)*r*r) + (distCoeffs.at <double>(0,4)*r*r*r));
                  xD = ((2*distCoeffs.at <double>(0,2)*y*x) + (distCoeffs.at <double>(0,3)*(r + (2*x*x))));
                  yD = ((2*distCoeffs.at <double>(0,3)*x*y) + (distCoeffs.at <double>(0,2)*(r + (2*y*y))));
                  xa = x; ya = y;
                  x = (x0 - xD)*idistra;
                  y = (y0 - yD)*idistra;
                  e = pow(xa - x,2)+ pow(ya - y,2);
                }*/

                corners[j + (i*patternsize.width)].x = x*cameraMatrix.at <double>(0,0) + cameraMatrix.at <double>(0,2);
                corners[j + (i*patternsize.width)].y = y*cameraMatrix.at <double>(1,1) + cameraMatrix.at <double>(1,2);


                //t = lambda.at <double>(2,2); // * Mat(Point3f(corners[j + (i*patternsize.width)].x, corners[j + (i*patternsize.width)].y, 1));
                //std::cout << t << '\n';
                xm += corners[j + (i*patternsize.width)].x;
                ym += corners[j + (i*patternsize.width)].y;
                //dist[j - 1] = euclideanDist(corners[j + (i*patternsize.width)], corners[(j - 1) + (i*patternsize.width)]);
                //distP += dist[j - 1];
              }
              xm /= (patternsize.width);
              ym /= (patternsize.width);
              s1 = 0;
              s2 = 0;
              s3 = 0;
              for( int j = 0; j < patternsize.width; ++j ){
                //errorRMS += (distP - dist[j - 1])*(distP - dist[j - 1]);
                s1 += (corners[j + (i*patternsize.width)].x - xm)*(corners[j + (i*patternsize.width)].y - ym);
                s2 += (corners[j + (i*patternsize.width)].x - xm)*(corners[j + (i*patternsize.width)].x - xm);
                s3 += (corners[j + (i*patternsize.width)].y - ym)*(corners[j + (i*patternsize.width)].y - ym);
              }
              if(s3 > s2){
                m = s1/s3;
                b = xm - (m*ym);
                for( int j = 0; j < patternsize.width; ++j ){
                  x = (m * corners[j + (i*patternsize.width)].y) + b;
                  x -= corners[j + (i*patternsize.width)].x;
                  if (m > 0.01) {
                    y = (corners[j + (i*patternsize.width)].x - b) / m;
                    y -= corners[j + (i*patternsize.width)].y;
                    x *= x;
                    y *= y;
                    error += sqrt(x * y)/sqrt(x + y);
                  } else{
                    x *= x;
                    error += x;
                  }
                }
              }else{
                m = s1/s2;
                b = ym - (m*xm);
                for( int j = 0; j < patternsize.width; ++j ){
                  y = (m * corners[j + (i*patternsize.width)].x) + b;
                  y -= corners[j + (i*patternsize.width)].y;
                  if (m > 0.01) {
                    x = (corners[j + (i*patternsize.width)].y - b) / m;
                    x -= corners[j + (i*patternsize.width)].x;
                    x *= x;
                    y *= y;
                    error += sqrt(x * y)/sqrt(x + y);
                  } else{
                    y *= y;
                    error += y;
                  }
                }
              }
            }
            
            error = (error) / ((patternsize.width)*patternsize.height) ;
            AcErr += error;
            drawChessboardCorners(view, patternsize, Mat(corners), patternfound);
            //std::cout << "error de linealidad = "<< error << '\n';
            numcap ++;
            canE ++;
          }

          imshow("Image Normal", view);
          imshow("Image Modif", rview);
          //video.write(rview);
          //flag = 1;
        }
      }
        char c = (char)waitKey(25);
        if( c  == 27 || imagePoints.size() == 25)
            break;
        else if(c == 'p'){
          flag = 1 - flag;
        }
        else if (numcap % 60 == 0 && patternfound) {
          imagePoints.push_back(corners);
          it++;
          std::cout << "capr #"<< it << '\n';
          //video.write(frame);
          flag2 = 1;
          patternfound = 0;
        }
        //error = AcErr/canE;
        //std::cout << "error RMS = "<< error << '\n';
    }
    capTest.release();
  }
  // Closes all the frames
  destroyAllWindows();

  return 0;
}
