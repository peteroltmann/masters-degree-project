#include "ContourEvolution.h"

#include "RegBasedContours.h"
#include "Selector.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>
#include <list>

using namespace cv;

ContourEvolution::ContourEvolution() {}

ContourEvolution::~ContourEvolution() {}

int ContourEvolution::run(std::string param_path)
{
    std::string image_path;
    Rect mask_rect;
    int iterations;
    int method;
    bool localized;
    int rad;
    float alpha;
    cv::Vec3f a;
    bool select_start_rect;
    bool no_sfm;

    // takes the data type's default value, if not set in file
    cv::FileStorage fs(param_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error opening '" << param_path << "'" << std::endl;
        std::cerr << "Specify parameterization file as argument or use "
                     "default: '../parameterization.yml'" << std::endl;
        return EXIT_FAILURE;
    }

    fs["image_path"] >> image_path;
    fs["mask_rect"] >> mask_rect;
    fs["iterations"] >> iterations;
    fs["method"] >> method;
    fs["localized"] >> localized;
    fs["rad"] >> rad;
    fs["alpha"] >> alpha;
    fs["a"] >> a;
    fs["select_start_rect"] >> select_start_rect;
    fs["no_sfm"] >> no_sfm;

    Mat frame;
    frame = imread(image_path, CV_LOAD_IMAGE_COLOR);
    namedWindow(WINDOW, WINDOW_AUTOSIZE);

    // check parameters
    if (frame.empty())
    {
        std::cerr << "Error loading image: '" << image_path << "'" << std::endl;
        return EXIT_FAILURE;
    }
    if (!select_start_rect && mask_rect == Rect(0, 0, 0, 0))
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
    if (alpha <= 0 || alpha >= 1)
    {
        alpha = .2f;
    }
    if (a == cv::Vec3f(0, 0, 0))
    {
        a = cv::Vec3f(1.f/3.f, 1.f/3.f, 1.f/3.f);
    }

    if (select_start_rect)
    {
        Selector selector(WINDOW, frame);
        cv::imshow(WINDOW, frame);
        while (!selector.is_valid())
        {
            cv::waitKey();
            if (!selector.is_valid())
            {
                std::cerr << "Invalid selection: " << selector.get_selection()
                          << std::endl;
            }
        }
        mask_rect = selector.get_selection();
    }

    cv::cvtColor(frame, frame, CV_RGB2GRAY);
    Mat mask = Mat::zeros(frame.size(), CV_8U);
    Mat roi(mask, mask_rect);
    roi = Scalar::all(1);

    if (localized)
    {
        // make image smaller for faster computation
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
        cv::resize(mask, mask, cv::Size(mask.cols/2, mask.rows/2));
    }

    RegBasedContours segm(Method(method), localized, rad, alpha);

#ifdef TIME_MEASUREMENT_TOTAL
    int64 t1, t2;
    t1 = cv::getTickCount();
#endif

    if (no_sfm)
        segm.apply(frame, mask, iterations);
    else
        segm.applySFM(frame, mask, iterations);


#ifdef TIME_MEASUREMENT_TOTAL
    t2 = cv::getTickCount();
    std::cout << "Total time [s]:" << (t2-t1)/cv::getTickFrequency()
              << std::endl;
#endif

    if (localized)
    {
        // make image smaller for faster computation
        cv::resize(segm.phi, segm.phi, cv::Size(frame.cols*2, frame.rows*2));
    }

    std::cout << "Done. Press key to show segmentation image." << std::endl;
    cv::waitKey();

    // get rid of eventual blobs
    cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
    inOut.setTo(255, segm.phi <= 0);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // show segmentation image
    Mat seg = Mat::zeros(frame.size(), CV_8U);
    cv::drawContours(seg, contours, -1, 255, CV_FILLED);
    imshow(WINDOW, seg);

    std::cout << "Done. Press key to quit." << std::endl;
    waitKey(0);

    return EXIT_SUCCESS;
}
