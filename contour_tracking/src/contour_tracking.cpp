#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "Contour.h"
#include "RegBasedContours.h"
#include "ContourParticleFilter.h"

#define WINDOW_NAME "Image"

void draw_contour(cv::Mat& frame, const cv::Mat_<uchar>& contour_mask)
{
    cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, frame.type());
    inOut.setTo(255, contour_mask == 1);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(frame, contours, -1, cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char *argv[])
{
    bool vrm;
    cv::Mat_<float> templ;
    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    cv::FileStorage fs2("../templ.yml", cv::FileStorage::READ);
//    cv::FileStorage fs2("../templ_walking.yml", cv::FileStorage::READ);
    fs["vrm"] >> vrm;
    fs2["templ"] >> templ;

    cv::namedWindow(WINDOW_NAME);
    int key = 0;
    cv::Mat frame;

    RegBasedContours segm;
    ContourParticleFilter pf(200);
    pf.init(templ);

    Contour templ_contour;
    templ.copyTo(templ_contour.contour_mask);

    // for this video: cv::cvtColor(frame, frame, CV_RGB2GRAY);
    cv::VideoCapture capture("../input/car_orig.avi");
//    cv::VideoCapture capture("../input/walking_2.avi");
//    cv::VideoCapture capture("../input/palau2_gray_cropped/palau2_frames_%04d.png");

    if (!capture.isOpened())
    {
        std::cerr << "Failed to open capture." << std::endl;
        return EXIT_FAILURE;
    }


    bool first_frame = true;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;

        capture >> frame;
        cv::cvtColor(frame, frame, CV_RGB2GRAY);

        if (first_frame)
        {
            segm.setFrame(frame);
            segm.init(templ);
            templ_contour.calc_energy(segm);
            std::cout << templ_contour.energy << std::endl;
            first_frame = false;
        }

        pf.predict();

        // DO TRANSFORM AND EVOLUTION: START FROM TEMPLATE
        // TODO: UPDATE EACH CONTOUR WITH EVOLUTION PREDICTION
        for (int i = 0; i < pf.num_particles; i++)
        {
            templ.copyTo(pf.pc[i]->contour_mask);
            pf.pc[i]->transform_affine(pf.p[i]);
            pf.pc[i]->evolve_contour(segm, frame, 10);
            pf.pc[i]->calc_energy(segm);
//            std::cout << pf.pc[i]->energy << std::endl;
//            pf.pc[i]->calc_distance(segm);
        }

        pf.calc_weight(templ_contour.energy);
//        for (int i = 0; i < pf.num_particles; i++)
//        {
//            std::cout << pf.w[i] << std::endl;
//        }
        pf.weighted_mean_estimate();

        // show contour
        templ.copyTo(pf.pc[0]->contour_mask);
        pf.pc[0]->transform_affine(pf.state);
        pf.pc[0]->evolve_contour(segm, frame, 10);
        draw_contour(frame, pf.pc[0]->contour_mask);
        cv::imshow(WINDOW_NAME, frame);
        key = cv::waitKey(500);

        pf.resample();

    }

    return EXIT_SUCCESS;
}
