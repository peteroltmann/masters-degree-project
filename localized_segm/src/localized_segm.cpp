#include "RegBasedContoursC3.h"
#include "Contour.h"

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
    int rad;
    float alpha;
    bool localized;

    // takes the data type's default value, if not set in file
    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    fs["imagePath"] >> imagePath;
    fs["maskRect"] >> maskRect;
    fs["iterations"] >> iterations;
    fs["method"] >> method;
    fs["localized"] >> localized;
    fs["rad"] >> rad;
    fs["alpha"] >> alpha;

    // check parameters
    Mat frame;
    frame = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
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

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

    RegBasedContoursC3 segm;
    Contour contour(mask);

#ifdef TIME_MEASUREMENT_TOTAL
    int64 t1, t2;
    t1 = cv::getTickCount();
#endif

//    segm.apply(frame, mask, phi, iterations, method, localized, rad, alpha);
    segm.applySFM(frame, mask, iterations, method, localized, rad, alpha);
//    contour.evolve(segm, frame, iterations);

#ifdef TIME_MEASUREMENT_TOTAL
    t2 = cv::getTickCount();
    std::cout << "Total time [s]:" << (t2-t1)/cv::getTickFrequency()
              << std::endl;
#endif

//    cv::FileStorage fs2("../templ.yml", cv::FileStorage::WRITE);
//    cv::Mat templ(frame.size(), CV_8U, cv::Scalar(0));
//    templ.setTo(1, segm._phi <= 0);
//    fs2 << "templ" << templ;

    if (localized)
    {
        // make image smaller for faster computation
        cv::resize(segm._phi, segm._phi, cv::Size(frame.cols*2, frame.rows*2));
    }

    cv::waitKey();

    // show segmentation image
    Mat seg = Mat::zeros(frame.size(), CV_8U);
    seg.setTo(255, segm._phi <= 0);
    imshow(WINDOW_NAME, seg);
//    imshow(WINDOW_NAME, contour.mask == 1);


    std::cout << "Done. Press key to quit." << std::endl;
    waitKey(0);

    return EXIT_SUCCESS;
}
