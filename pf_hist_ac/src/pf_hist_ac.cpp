#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

#include "RegBasedContours.h"
#include "ParticleFilter.h"
#include "StateParams.h"
#include "hist.h"

#define WHITE cv::Scalar(255, 255, 255)
#define BLUE cv::Scalar(255, 0, 0)
#define GREEN cv::Scalar(0, 255, 0)
#define RED cv::Scalar(0, 0, 255)

#define WINDOW_NAME "Image"
#define WINDOW_TEMPALTE_NAME "Template"
#define WINDOW_ROI_NAME "ROI"

void draw_contour(cv::Mat& window_image, const cv::Mat_<uchar>& contour_mask,
                  cv::Scalar color)
{
    cv::Mat inOut = cv::Mat::zeros(window_image.rows, window_image.cols, CV_8U);
    inOut.setTo(255, contour_mask == 1);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(window_image, contours, -1, color, 1);
}

int main(int argc, char *argv[])
{
    const int NUM_PARTICLES = 100;
    const int NUM_ITERATIONS = 200;
//    cv::Rect templ(245, 10, 90, 65);
    cv::Mat_<uchar> templ;
    cv::Mat_<uchar> templ_frame0;
    cv::Mat_<float> templ_hist;
    cv::Rect evolved_bound;
    cv::Rect templ_bound;

    cv::Mat_<float> hu1, hu2, hu4; // for matlab output

//    cv::FileStorage fs("../templ.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../templ_walking.yml", cv::FileStorage::READ);
    cv::FileStorage fs("../templ_fish.yml", cv::FileStorage::READ);
    fs["templ"] >> templ;

    int key = 0;
    cv::namedWindow(WINDOW_NAME);
    cv::namedWindow(WINDOW_TEMPALTE_NAME);
    cv::namedWindow(WINDOW_ROI_NAME, CV_WINDOW_NORMAL);
    cv::Mat frame;
    cv::Mat templ_image;
    cv::Mat window_image;
    cv::Mat window_templ_image;

    RegBasedContours segm;

    ParticleFilter pf(NUM_PARTICLES);
    pf.init(templ);

    // for this video: cv::cvtColor(frame, frame, CV_RGB2GRAY);
//    cv::VideoCapture capture("../input/car_orig.avi");
//    cv::VideoCapture capture("../input/walking_2.avi");
    cv::VideoCapture capture("../input/palau2_gray_cropped/palau2_frames_%04d.png");

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

    int cnt_frame = 0;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;

//        cv::rectangle(frame, templ, cv::Scalar(255, 255, 255), 1);
//        cv::imshow(WINDOW_NAME, frame);
//        cv::waitKey(0);

        if (templ_hist.empty())
        {
            templ.copyTo(templ_frame0);
            frame.copyTo(templ_image);
//            cv::Mat frame_roi(frame, templ);
            calc_hist(frame, templ_hist, templ);
//            cv::normalize(templ_hist, templ_hist);
            normalize(templ_hist);
        }

//        cv::cvtColor(frame, frame, CV_RGB2GRAY);
        cv::cvtColor(frame, window_image, CV_GRAY2BGR);

#ifndef PARTICLE_FILTER
        segm.applySFM(frame, contour, 50);
        contour.setTo(0);
        contour.setTo(1, segm._phi <= 0);
#endif // !PF_AC

#ifdef PARTICLE_FILTER

        pf.predict();
        pf.calc_weight(frame, templ, templ_hist);

/*
        for (int i = 0; i < pf.num_particles; i++)
        {
            std::cout << pf.w[i] << std::endl;
        }
*/

        cv::Rect bounds(0, 0, frame.cols, frame.rows);

/*
        // draw particles
        cv::Moments m = cv::moments(templ, true);
        cv::Point2f center(m.m10/m.m00, m.m01/m.m00);
        cv::Mat_<float> templ_at_zero = (cv::Mat_<float>(4, 1) <<
                                      -center.x, -center.y, 0, 0);
        for (int i = 0; i < pf.num_particles; i++)
        {
            cv::Mat_<float> tmp = templ_at_zero + pf.p[i];
            Contour pi_c;
            templ.copyTo(pi_c.contour_mask);
            pi_c.transform_affine(tmp);
            draw_contour(window_image, pi_c.contour_mask, RED);
        }
*/

        if (cnt_frame == 0)
            draw_contour(window_image, templ, cv::Scalar(255, 0, 0));
        draw_contour(window_image, pf.state_c.contour_mask, GREEN);

        // evolve contour
        Contour state_c_evolved;
        pf.state_c.contour_mask.copyTo(state_c_evolved.contour_mask);

        state_c_evolved.evolve_contour(segm, frame, NUM_ITERATIONS);
        draw_contour(window_image, state_c_evolved.contour_mask, WHITE);

        // get rid of eventual blobs
//        cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
//        inOut.setTo(255, state_c_evolved.contour_mask == 1);
//        std::vector< std::vector<cv::Point> > contours;
//        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//        state_c_evolved.contour_mask.setTo(0);
//        cv::drawContours(state_c_evolved.contour_mask, contours, -1, WHITE, CV_FILLED);
//        state_c_evolved.contour_mask.setTo(1, state_c_evolved.contour_mask == 255);

        // calculate bounding rectangles
        evolved_bound = bounding_rect(state_c_evolved.contour_mask);
        templ_bound = bounding_rect(templ);

        cv::rectangle(window_image, evolved_bound, GREEN);

        // after contour evolution:
        // check if template and template histogram can be adpated in
        // user-defined checking interval
        cv::Mat_<float> evolved_hist;
        calc_hist(frame, evolved_hist, state_c_evolved.contour_mask);
//        cv::normalize(evolved_hist, evolved_hist);
//        float bc_templ = cv::compareHist(templ_hist, evolved_hist,
//                                         CV_COMP_BHATTACHARYYA);
        normalize(evolved_hist);
        float bc_templ = calcBC(templ_hist, evolved_hist);

        cv::Mat_<uchar> templ_roi(templ, templ_bound);
        cv::Mat_<uchar> evolved_roi(state_c_evolved.contour_mask, evolved_bound);

        cv::Mat test = cv::Mat::zeros(evolved_roi.size(), evolved_roi.type());
        test.setTo(255, evolved_roi == 1);
        cv::imshow(WINDOW_ROI_NAME, test);

        float hu_templ_1 = match_shapes(templ_roi, evolved_roi,
                                        CV_CONTOURS_MATCH_I1);
        float hu_templ_2 = match_shapes(templ_roi, evolved_roi,
                                        CV_CONTOURS_MATCH_I2);
        float hu_templ_4 = match_shapes(templ_roi, evolved_roi, 4);

        hu1.push_back(hu_templ_1);
        hu2.push_back(hu_templ_2);
        hu4.push_back(hu_templ_4);


        // TODO histogram adpation instead of replacement?
        //      consider more frames that keep this status
/*
        if ((cnt_frame % 19 == 0 && cnt_frame != 0) &&
            (hu_templ < .15f && bc_templ < .35f))
        {
            state_c_evolved.contour_mask.copyTo(templ);
            calc_hist(frame, templ_hist, templ);
//            cv::normalize(templ_hist, templ_hist);
            normalize(templ_hist);
//            frame.copyTo(templ_image);
            cv::cvtColor(frame, templ_image, CV_GRAY2BGR);
        }
*/

/*
        if ((bc_templ >= 3.f || hu_templ >= .20f) ||
            (bc_templ >= .18f && hu_templ >= .1f))
        {
            cv::rectangle(window_image, cv::Rect(5, 5, 20, 20), RED, -1);
        }
*/


        // update state and resampling
        pf.weighted_mean_estimate();
        pf.resample();
//        pf.resample_systematic();


//        cv::imshow("Template Histogram", draw_hist(templ_hist));
//        cv::imshow("Estimate Histogram", draw_hist(estimate_hist));
//        cv::imshow("Evolved Histogram", draw_hist(evolved_hist));



        // draw template contour
        cv::cvtColor(templ_image, window_templ_image, CV_GRAY2BGR);
//        templ_image.copyTo(window_templ_image);
        draw_contour(window_templ_image, templ, BLUE);
        cv::rectangle(window_templ_image, templ_bound, GREEN);


        float hu_diff = hu_templ_2 - hu_templ_1;
        // output data
        std::cout << boost::format("#%03d: bc-templ[%f], hu-templ-1[%f], hu-templ-2[%f], hu-templ-4[%f]")
                     % cnt_frame % bc_templ % hu_templ_1 % hu_templ_2 % hu_templ_4;

        // test occlusion
        if (cnt_frame > 125 && cnt_frame < 185) // Verdeckung
        {
            std::cout << " - Occlusion";
        }

        // test deformation (directly after)
        if (cnt_frame >= 185 && cnt_frame < 220)
        {
            std::cout << " - Deformation";
        }
        if (cnt_frame >= 285 && cnt_frame < 330)
        {
            std::cout << " - Deformation";
        }

        std::cout << std::endl;

#else
        draw_contour(frame, contour);
#endif // PF_AC

        cv::imshow(WINDOW_NAME, window_image);
        cv::imshow(WINDOW_TEMPALTE_NAME, window_templ_image);
        key = cv::waitKey(1);

        // pause on space
        if (key == ' ')
            key = cv::waitKey(0);

#ifdef SAVE_VIDEO
        videoOut << frame;
#endif

        cnt_frame++;
    }

    std::ofstream hu_output("C:/Users/Peter/Documents/MATLAB/humatch.m");
    hu_output << "hu1 = " << hu1 << ";" << std::endl;
    hu_output << "hu2 = " << hu2 << ";" << std::endl;
    hu_output << "hu4 = " << hu4 << ";" << std::endl;
    hu_output.close();

#ifdef SAVE_VIDEO
    videoOut.release();
#endif
    return EXIT_SUCCESS;
}
