#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

#define WINDOW "Image"

using namespace cv;

int main(int argc, char** argv)
{
    namedWindow(WINDOW);
    Mat frame;
    frame = imread("../src/airplane.jpg",
                       CV_LOAD_IMAGE_GRAYSCALE);
    Mat seg = Mat::zeros(frame.size(), CV_8U);
    Mat mask = Mat::zeros(frame.size(), frame.type());
    Mat roi(mask, Rect(122, 110, 112, 112)); // from matlab demo
    roi = Scalar::all(1);

    // make image smaller for fast computation
//    cv::resize(frame, frame, cv::Size(frame.cols*0.5, frame.rows*0.5));
//    cv::resize(mask, mask, cv::Size(mask.cols*0.5, mask.rows*0.5));

    RegBasedContours segm;
    segm.apply(frame, mask, seg, 600);

    imshow(WINDOW, seg);
    std::cout << "Done. Press key to quit." << std::endl;
    waitKey(0);

    return EXIT_SUCCESS;
}
