#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    std::string imagePath;
    Rect maskRect;
    int iterations;
    int method;
    int rad;
    float alpha;
    bool localized;

    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    fs["imagePath"] >> imagePath;
    fs["maskRect"] >> maskRect;
    fs["iterations"] >> iterations;
    fs["method"] >> method;
    fs["localized"] >> localized;
    fs["rad"] >> rad;
    fs["alpha"] >> alpha;

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
    Mat frame;
    frame = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
    if (frame.empty())
    {
        std::cerr << "Error loading image: '" << imagePath << "'" << std::endl;
        return EXIT_FAILURE;
    }

    Mat phi;
    Mat mask = Mat::zeros(frame.size(), frame.type());
    Mat roi(mask, maskRect);
    roi = Scalar::all(1);

    // make image smaller for fast computation
    cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
    cv::resize(mask, mask, cv::Size(mask.cols/2, mask.rows/2));

    RegBasedContours segm;
    segm.apply(frame, mask, phi, iterations, method, localized, rad, alpha);

    // show segmentation image
    Mat seg = Mat::zeros(frame.size(), CV_8U);
    seg.setTo(255, phi < 0);
    imshow(WINDOW_NAME, seg);

    std::cout << "Done. Press key to quit." << std::endl;
    waitKey(0);

    return EXIT_SUCCESS;
}
