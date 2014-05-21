#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "VRmUsbCam.h"

#define WINDOW_NAME "Image"

int main(int argc, char** argv)
{

    VRmUsbCam capture;
    if (!capture.open())
    {
        return EXIT_FAILURE;
    }

    cv::namedWindow(WINDOW_NAME);
    int key = 0;
    cv::Mat frame;

    while (key != 'q')
    {
        capture >> frame;
        cv::imshow(WINDOW_NAME, frame);
        key = cv::waitKey(1);
    }

    capture.close();

    return EXIT_SUCCESS;
}
