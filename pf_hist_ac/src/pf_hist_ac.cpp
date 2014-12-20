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
    // see 'parameterization.yml' for description
    int num_particles;
    int num_iterations;
    float sigma;
    std::string templ_path;
    bool cvt_color;
    std::string input_path;
    bool save_video;
    double fps;
    std::string output_path;
    bool save_img_seq;
    std::string save_img_path;

    cv::FileStorage fs("../parameterization.yml", cv::FileStorage::READ);
    fs["num_particles"] >> num_particles;
    fs["num_iterations"] >> num_iterations;
    fs["sigma"] >> sigma;
    fs["templ_path"] >> templ_path;
    fs["cvt_color"] >> cvt_color;
    fs["input_path"] >> input_path;
    fs["save_video"] >> save_video;
    fs["fps"] >> fps;
    fs["output_path"] >> output_path;
    fs["save_img_seq"] >> save_img_seq;
    fs["save_img_path"] >> save_img_path;

    cv::FileStorage fs2(templ_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error loading template: '" << templ_path << "'"
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (num_particles <= 0)
    {
        std::cout << "invalid value for 'num_particles', '100' used instead"
                  << std::endl;
        num_particles = 100;
    }

    if (num_iterations <= 0)
    {
        std::cout << "invalid value for 'num_iterations', '10' used instead"
                  << std::endl;
        num_particles = 10;
    }

    if (sigma <= 0.f)
    {
        std::cout << "invalid value for 'sigma', '25.0' used instead"
                  << std::endl;
        sigma = 20.f;
    }



//    cv::FileStorage fs("../templ.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../templ_walking.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../templ_fish.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../templ_aircraft.yml", cv::FileStorage::READ);
//    cv::FileStorage fs("../templ_aircraft_big.yml", cv::FileStorage::READ);

    // for this video: cv::cvtColor(frame, frame, CV_RGB2GRAY);
//    cv::VideoCapture capture("../input/car_orig.avi");
//    cv::VideoCapture capture("../input/walking_2.avi");
//    cv::VideoCapture capture("../input/palau2_gray_cropped/palau2_frames_%04d.png");
//    cv::VideoCapture capture("../input/aerobatics_1_3.avi");
//    cv::VideoCapture capture("../input/big_1_3.avi");

    int key = 0;
    cv::namedWindow(WINDOW_NAME);
    cv::namedWindow(WINDOW_TEMPALTE_NAME);
//    cv::namedWindow(WINDOW_ROI_NAME, CV_WINDOW_NORMAL);
    cv::Mat frame;

    cv::Mat templ_image;
    cv::Mat window_image;
    cv::Mat window_templ_image;

    cv::Mat_<uchar> templ;
    fs2["templ"] >> templ;
    cv::Rect templ_rect = bounding_rect(templ);
    cv::Mat_<uchar> templ_frame0;
    cv::Mat_<float> templ_hist;

    cv::Rect evolved_bound;
    cv::Rect templ_bound;

    cv::Mat_<float> hu1; // for matlab output

    RegBasedContours segm;
    ParticleFilter pf(num_particles);
    pf.init(templ_rect);

    cv::VideoCapture capture(input_path);

    if (!capture.isOpened())
    {
        std::cerr << "Failed to open capture." << std::endl;
        return EXIT_FAILURE;
    }

    cv::VideoWriter videoOut;
    if(save_video)
    {
        videoOut.open(output_path, -1, fps, frame.size(), false);
        if (!videoOut.isOpened())
        {
            std::cerr << "Could not write output video" << std::endl;
            return EXIT_FAILURE;
        }
    }

    int cnt_frame = 0;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;

        if (cvt_color)
            cv::cvtColor(frame, frame, CV_RGB2GRAY);

        cv::cvtColor(frame, window_image, CV_GRAY2BGR);

        cv::Rect bounds(0, 0, frame.cols, frame.rows);

        // calc template histogram on first frame
        if (templ_hist.empty())
        {

            templ.copyTo(templ_frame0);
            frame.copyTo(templ_image);
            cv::cvtColor(frame, window_templ_image, CV_GRAY2BGR);

            cv::Mat frame_roi(frame, templ_rect);
            calc_hist(frame, templ_hist, templ);
            normalize(templ_hist);
        }

        // =====================================================================
        // = PARTICLE FILTER                                                   =
        // =====================================================================

        pf.predict();
        pf.calc_weight(frame, templ_rect.size(), templ_hist, sigma);

/*
        for (int i = 0; i < pf.num_particles; i++)
        {
            std::cout << pf.w[i] << std::endl;
        }
*/

/*
        // draw particles
        for (int i = 0; i < pf.num_particles; i++)
        {
            int x = std::round(pf.p[i](PARAM_X)) - templ_rect.width/2;
            int y = std::round(pf.p[i](PARAM_Y)) - templ_rect.height/2;
            cv::Rect state_rect = cv::Rect(x, y, templ_rect.width,
                                           templ_rect.height) & bounds;
            cv::rectangle(window_image, state_rect, RED, 1);
        }
*/

        // draw predicted estimate
        int width = round(templ_rect.width * pf.state(PARAM_SCALE));
        int height = round(templ_rect.height * pf.state(PARAM_SCALE));
        int x = std::round(pf.state(PARAM_X)) - width/2;
        int y = std::round(pf.state(PARAM_Y)) - height/2;
        cv::Rect state_rect = cv::Rect(x, y, width, height) & bounds;

        cv::rectangle(window_image, state_rect, GREEN, 1);


        // =====================================================================
        // = CONTOUR EVOLUTION
        // =====================================================================

        // init contour with state rectangle
        Contour state_c_evolved;
        state_c_evolved.contour_mask = cv::Mat_<uchar>::zeros(frame.size());
        cv::Mat state_roi(state_c_evolved.contour_mask, state_rect);
        state_roi.setTo(1);

        // evolve contour
        state_c_evolved.evolve_contour(segm, frame, num_iterations);
        draw_contour(window_image, state_c_evolved.contour_mask, WHITE);


        // get rid of eventual blobs
//        cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
//        inOut.setTo(255, state_c_evolved.contour_mask == 1);
//        std::vector< std::vector<cv::Point> > contours;
//        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//        state_c_evolved.contour_mask.setTo(0);
//        cv::drawContours(state_c_evolved.contour_mask, contours, -1, WHITE, CV_FILLED);
//        state_c_evolved.contour_mask.setTo(1, state_c_evolved.contour_mask == 255);

        // =====================================================================
        // = AFTER CONTOUR EVOLUTION                                           =
        // =====================================================================

        // draw template contour
        draw_contour(window_templ_image, templ, BLUE);

        // calculate bounding rectangles
        evolved_bound = bounding_rect(state_c_evolved.contour_mask);
        templ_bound = bounding_rect(templ);
        cv::rectangle(window_image, evolved_bound, WHITE);
        cv::rectangle(window_templ_image, templ_bound, BLUE);

        cv::Mat_<uchar> templ_roi(templ, templ_bound);
        cv::Mat_<uchar> evolved_roi(state_c_evolved.contour_mask, evolved_bound);

/* =============================================================================

        // check if template and template histogram can be adpated in
        // user-defined checking interval
        cv::Mat_<float> evolved_hist;
        calc_hist(frame, evolved_hist);
        normalize(evolved_hist);
        float bc_templ = calcBC(templ_hist, evolved_hist);



//        cv::Mat test = cv::Mat::zeros(evolved_roi.size(), evolved_roi.type());
//        test.setTo(255, evolved_roi == 1);
//        cv::imshow(WINDOW_ROI_NAME, test);

        float hu_templ_1 = match_shapes(templ_roi, evolved_roi,
                                        CV_CONTOURS_MATCH_I1);
        hu1.push_back(hu_templ_1);

        // detect successful tracking, so template can be adapted
//        if (bc_templ < 0.1) // just try out
//        {
//            state_c_evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
//            normalize(templ_hist);
//            frame.copyTo(templ_image);
//        }


        // TODO histogram adpation instead of replacement?
        //      consider more frames that keep this status

//        if ((cnt_frame % 19 == 0 && cnt_frame != 0) &&
//            (hu_templ < .15f && bc_templ < .35f))
//        {
//            state_c_evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
////            cv::normalize(templ_hist, templ_hist);
//            normalize(templ_hist);
////            frame.copyTo(templ_image);
//            cv::cvtColor(frame, templ_image, CV_GRAY2BGR);
//        }

//        bool lost = bc_templ > 0.2; // detected as lost
//        if (lost)
//        {
//            cv::rectangle(window_image, cv::Rect(5, 5, 20, 20), RED, -1);
//            state_c_evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
//            normalize(templ_hist);
//            cv::cvtColor(frame, templ_image, CV_GRAY2BGR);
//        }


//        cv::imshow("Template Histogram", draw_hist(templ_hist));
//        cv::imshow("Estimate Histogram", draw_hist(estimate_hist));
//        cv::imshow("Evolved Histogram", draw_hist(evolved_hist));


        // output data
        std::cout << boost::format("#%03d: bc-templ[%f], hu-templ-1[%f]")
                     % cnt_frame % bc_templ % hu_templ_1;

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

============================================================================= */

        // =====================================================================
        // = UPDATE PARTICLE FILTER                                            =
        // =====================================================================

        // update state and resampling
        pf.weighted_mean_estimate();
        pf.resample();
//        pf.resample_systematic();

        // =====================================================================
        // = IMAGE OUTPUT                                                      =
        // =====================================================================

        cv::imshow(WINDOW_NAME, window_image);
        cv::imshow(WINDOW_TEMPALTE_NAME, window_templ_image);
        key = cv::waitKey(1);

        if (save_img_seq)
        {
            std::stringstream s;
            s << boost::format(save_img_path) % cnt_frame;
            cv::imwrite(s.str(), window_image);
        }

        if (save_video)
            videoOut << window_image;

        // pause on space
        if (key == ' ')
            key = cv::waitKey(0);

        cnt_frame++;
    }

    // =========================================================================
    // = HU OUTPUT (MATLAB)                                                    =
    // =========================================================================

    std::ofstream hu_output("C:/Users/Peter/Documents/MATLAB/humatch.m");
    hu_output << "hu1 = " << hu1 << ";" << std::endl;
    hu_output.close();

    if (save_video)
        videoOut.release();
    return EXIT_SUCCESS;
}
