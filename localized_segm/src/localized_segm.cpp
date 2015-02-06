#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>
#include <list>

using namespace cv;

int main(int argc, char** argv)
{
    std::string imagePath;
    Rect maskRect;
    int iterations;
    int method;
    bool localized;
    int rad;
    float alpha;
    cv::Vec3f a;

    // takes the data type's default value, if not set in file
    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    fs["imagePath"] >> imagePath;
    fs["maskRect"] >> maskRect;
    fs["iterations"] >> iterations;
    fs["method"] >> method;
    fs["localized"] >> localized;
    fs["rad"] >> rad;
    fs["alpha"] >> alpha;
    fs["a"] >> a;

    // check parameters
    Mat frame;
    frame = imread(imagePath, CV_LOAD_IMAGE_COLOR);
//    cv::cvtColor(frame, frame, CV_RGB2HSV);
    cv::cvtColor(frame, frame, CV_RGB2GRAY);
    if (frame.empty())
    {
        std::cerr << "Error loading image: '" << imagePath << "'" << std::endl;
        return EXIT_FAILURE;
    }
    if (maskRect == Rect(0, 0, 0, 0))
    {
        std::cerr << "No initialisation mask given" << std::endl;
        return EXIT_FAILURE;
    }
    if (iterations <= 0)
    {
        std::cerr << "Invalid number of iterations: " << iterations
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (method != CHAN_VESE && method != YEZZI)
    {
        std::cerr << "Invalid method: " << method << std::endl;
        return EXIT_FAILURE;
    }
    if (rad <= 0)
    {
        // default radius dependent on frame size
        rad = std::round((frame.rows+frame.cols)/(2*8));
    }
    if (alpha <= 0)
    {
        alpha = .2f;
    }
    if (a == cv::Vec3f(0, 0, 0))
    {
        a = cv::Vec3f(1.f/3.f, 1.f/3.f, 1.f/3.f);
    }

    Mat phi;
    Mat mask = Mat::zeros(frame.size(), CV_8U);
    Mat roi(mask, maskRect);
    roi = Scalar::all(1);

    if (localized)
    {
        // make image smaller for faster computation
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
        cv::resize(mask, mask, cv::Size(mask.cols/2, mask.rows/2));
    }

    namedWindow(WINDOW, WINDOW_AUTOSIZE);

    RegBasedContours segm;

#ifdef TIME_MEASUREMENT_TOTAL
    int64 t1, t2;
    t1 = cv::getTickCount();
#endif

    segm.applySFM(frame, mask, iterations, method, localized, rad, alpha);
//    segm.applySFM(frame, mask, iterations, method, localized, rad, alpha, a);

#ifdef TIME_MEASUREMENT_TOTAL
    t2 = cv::getTickCount();
    std::cout << "Total time [s]:" << (t2-t1)/cv::getTickFrequency()
              << std::endl;
#endif

    if (localized)
    {
        // make image smaller for faster computation
        cv::resize(segm._phi, segm._phi, cv::Size(frame.cols*2, frame.rows*2));
    }

    std::cout << "Done. Press key to show segmentation image." << std::endl;
    cv::waitKey();

    // get rid of eventual blobs
    cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
    inOut.setTo(255, segm._phi <= 0);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // show segmentation image
    Mat seg = Mat::zeros(frame.size(), CV_8U);
    cv::drawContours(seg, contours, -1, 255, CV_FILLED);
    imshow(WINDOW, seg);

//    cv::FileStorage fs2("../templ.yml", cv::FileStorage::WRITE);
//    cv::Mat templ(frame.size(), CV_8U, cv::Scalar(0));
//    templ.setTo(1, seg == 255);
//    fs2 << "templ" << templ;

    std::cout << "Done. Press key to quit." << std::endl;
    waitKey(0);

    return EXIT_SUCCESS;
}
