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
//    cv::FileStorage fs2("../templ_plane.yml", cv::FileStorage::READ);
    fs["vrm"] >> vrm;
    fs2["templ"] >> templ;

    const int NUM_PARTICLES = 100;
    const int NUM_ITERATIONS = 10;

    cv::namedWindow(WINDOW_NAME);
    int key = 0;
    cv::Mat frame;

    RegBasedContours segm;
    ContourParticleFilter pf(NUM_PARTICLES);
    pf.init(templ);

    Contour templ_contour;
    templ.copyTo(templ_contour.contour_mask);

    // for this video: cv::cvtColor(frame, frame, CV_RGB2GRAY);
    cv::VideoCapture capture("../input/car_orig.avi");
//    cv::VideoCapture capture("../input/walking_2.avi");
//    cv::VideoCapture capture("../input/PlaneSequence/frame_00013451.pgm");
//    cv::VideoCapture capture("../input/palau2_gray_cropped/palau2_frames_%04d.png");

    if (!capture.isOpened())
    {
        std::cerr << "Failed to open capture." << std::endl;
        return EXIT_FAILURE;
    }

#ifdef SAVE_VIDEO
#define VIDEO_FILE "C:/Users/Peter/Desktop/output.avi"
    cv::VideoWriter videoOut;
    videoOut.open(VIDEO_FILE, -1, 15, frame.size(), false);
    if (!videoOut.isOpened())
    {
        std::cerr << "Could not write output video" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    bool first_frame = true;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, frame, CV_RGB2GRAY);

#ifndef PF_AC
        templ_contour.evolve_contour(segm, frame, 50);
#endif // !PF_AC

#ifdef PF_AC

        if (first_frame)
        {
            segm.setFrame(frame);
            segm.init(templ);
//            segm.iterate();
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
//            draw_contour(frame, pf.pc[i]->contour_mask);
//            cv::imshow(WINDOW_NAME, frame);
//            key = cv::waitKey(0);
            pf.pc[i]->evolve_contour(segm, frame, NUM_ITERATIONS);
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
/*
        for (int i = 0; i < pf.num_particles; i++)
        {
            templ.copyTo(pf.pc[0]->contour_mask);
//            pf.pc[0]->transform_affine(pf.state);
//            pf.pc[0]->evolve_contour(segm, frame, NUM_ITERATIONS);
            draw_contour(frame, pf.pc[0]->contour_mask);
            cv::imshow(WINDOW_NAME, frame);
            key = cv::waitKey(1);
        }
*/

        templ.copyTo(pf.pc[0]->contour_mask);
        pf.pc[0]->transform_affine(pf.state);
        // The "trick": more evolution steps, usually: NUM_ITERATIONS
        pf.pc[0]->evolve_contour(segm, frame, NUM_ITERATIONS);
        draw_contour(frame, pf.pc[0]->contour_mask);
//        draw_contour(frame, pf.state_c.contour_mask);


//        pf.resample();
        pf.resample_systematic();

#else
        draw_contour(frame, templ_contour.contour_mask);
#endif // PF_AC

        cv::imshow(WINDOW_NAME, frame);
        key = cv::waitKey(1);

#ifdef SAVE_VIDEO
        videoOut << frame;
#endif

    }
#ifdef SAVE_VIDEO
    videoOut.release();
#endif
    return EXIT_SUCCESS;
}
