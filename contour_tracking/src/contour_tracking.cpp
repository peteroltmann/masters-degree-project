#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "VRmUsbCam.h"

#define WINDOW_NAME "Image"

int main(int argc, char** argv)
{
    bool vrm;
    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    fs["vrm"] >> vrm;

    cv::VideoCapture* capture;
    if (vrm)
        capture = new VRmUsbCam();
    else
        capture = new cv::VideoCapture();
    if (!capture->open(0))
    {
        return EXIT_FAILURE;
    }

    cv::namedWindow(WINDOW_NAME);
    int key = 0;
    cv::Mat frame;

    while (key != 'q')
    {
        *capture >> frame;
        cv::imshow(WINDOW_NAME, frame);
        key = cv::waitKey(1);
    }

    capture->release();
    delete capture;

    return EXIT_SUCCESS;
}
