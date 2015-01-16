#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

#include "RegBasedContours.h"
#include "ParticleFilter.h"
#include "StateParams.h"
#include "Histogram.h"
#include "FourierDescriptor.h"

#define WHITE cv::Scalar(255, 255, 255)
#define BLUE cv::Scalar(255, 0, 0)
#define GREEN cv::Scalar(0, 255, 0)
#define RED cv::Scalar(0, 0, 255)

#define WINDOW_NAME "Image"
#define WINDOW_TEMPALTE_NAME "Template"
#define WINDOW_SEG "ROI"

int main(int argc, char *argv[])
{
    // =========================================================================
    // = PARAMETERIZATION                                                      =
    // =========================================================================

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
    std::string matlab_file_path;

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
    fs["matlab_file_path"] >> matlab_file_path;

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
        std::cout << "invalid value for 'sigma', '20.0' used instead"
                  << std::endl;
        sigma = 20.f;
    }

    // =========================================================================
    // = DECLARATION AND INITIALIZATION                                        =
    // =========================================================================

    // windows
    int key = 0;
    cv::namedWindow(WINDOW_NAME);
    cv::namedWindow(WINDOW_TEMPALTE_NAME);
//    cv::namedWindow(WINDOW_SEG, CV_WINDOW_NORMAL);

    // images
    cv::Mat frame;
    cv::Mat frame_gray;
    cv::Mat templ_image;
    cv::Mat window_image;
    cv::Mat window_templ_image;

    // contour and histograms
    cv::Mat_<uchar> in;
    fs2["templ"] >> in;
    Contour templ(in);
    Contour templ_frame0(in);
    Histogram templ_hist;
    FourierDescriptor templ_fd(templ.mask, 64);

    cv::Mat_<float> hu1; // hu values for matlab output
    cv::Mat_<float> fd1; // fd values for matlab output

    RegBasedContours segm; // object prividing the contour evolution algorithm

    // init particle filter
    ParticleFilter pf(num_particles);
    pf.init(templ.bound);

    // open input image sequence or video
    cv::VideoCapture capture(input_path);
    if (!capture.isOpened())
    {
        std::cerr << "Failed to open capture." << std::endl;
        return EXIT_FAILURE;
    }

    cv::VideoWriter videoOut;
    if(save_video)
    {
        double input_fps = capture.get(CV_CAP_PROP_FPS);
        if (fps <= 0)
        {
            fps = input_fps;
        }

        videoOut.open(output_path, CV_FOURCC('X', 'V', 'I', 'D'), fps,
                      templ.mask.size(), true);
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
        if (frame.empty()) // end of image sequence
            break;

        cv::cvtColor(frame, frame_gray, CV_RGB2GRAY);
        frame.copyTo(window_image);

        cv::Rect bounds(0, 0, frame.cols, frame.rows);

        // calc template histogram on first frame
        if (templ_hist.empty())
        {
            // calc template histogram
            templ_hist.calc_hist(frame, RGB, templ.mask);

            // save frame as template image
            frame.copyTo(templ_image);
            frame.copyTo(window_templ_image);

            // draw template contour
            templ.draw(window_templ_image, BLUE);
            cv::rectangle(window_templ_image, templ.bound, BLUE);

        }

        // =====================================================================
        // = TRYOUT                                                            =
        // =====================================================================
/*
        FourierDescriptor fd(templ.mask, 64);

        float a = 0.785398163;

        cv::Mat Tm = (cv::Mat_<float>(3, 3) <<  1, 0,  -fd.center.x,
                                                0, 1,  -fd.center.y,
                                                0, 0,             1);

        cv::Mat S  = (cv::Mat_<float>(3, 3) <<  1.5,   0, 0,
                                                  0, 1.5, 0,
                                                  0,   0, 1);

        cv::Mat T  = (cv::Mat_<float>(3, 3) <<  1, 0,   fd.center.x,
                                                0, 1,   fd.center.y,
                                                0, 0,             1);

        cv::Mat R = (cv::Mat_<float>(3, 3) <<  cos(a), -sin(a), 0,
                                               sin(a),  cos(a), 0,
                                                    0,      0,  1);

        cv::Mat M = T*R*S*Tm;
        M.pop_back();
        std::cout << M << std::endl;

//        cv::Mat M = (cv::Mat_<float>(2, 3) <<  1, 0, -50,
//                                               0, 1, -50);

        cv::imshow("ASD", templ.mask == 1);
        cv::warpAffine(templ.mask, templ.mask, M, templ.mask.size());
//        templ.mask.setTo(0);
//        cv::rectangle(templ.mask, cv::Rect(100, 100, 100, 100), 1, -1);
        cv::imshow("ASD2", templ.mask == 1);
        cv::waitKey();

        FourierDescriptor fd2(templ.mask, 64);

        float match_fd = fd.match(fd2);
        std::cout << match_fd << std::endl;

        // TODO reconstruct with different amounts of fourier coefficients


        return EXIT_SUCCESS; // ################################################
*/
        // =====================================================================
        // = PARTICLE FILTER                                                   =
        // =====================================================================

        pf.predict();
        pf.calc_weight(frame, templ.bound.size(), templ_hist, sigma);

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
            cv::Rect pi_rect = pf.state_rect(templ.bound.size(), bounds, i);
            cv::rectangle(window_image, pi_rect, RED, 1);
        }
*/

        // get predicted estimate rectangle
        cv::Rect estimate_rect = pf.state_rect(templ.bound.size(), bounds);

        // draw predicted estimate
        cv::rectangle(window_image, estimate_rect, GREEN, 1);

        // =====================================================================
        // = CONTOUR EVOLUTION
        // =====================================================================

        // create init mask from estimated state rectangle
        cv::Mat init_mask = cv::Mat_<uchar>::zeros(frame.size());
        cv::Mat state_roi(init_mask, estimate_rect);
        state_roi.setTo(1);

        // evolve contour
        Contour evolved(init_mask);
        evolved.evolve(segm, frame_gray, num_iterations);
        evolved.draw(window_image, WHITE);
        cv::rectangle(window_image, evolved.bound, WHITE);

        // =====================================================================
        // = AFTER CONTOUR EVOLUTION                                           =
        // =====================================================================

        // calc evolved contour histogram
        Histogram evolved_hist;
        evolved_hist.calc_hist(frame, RGB, evolved.mask);

        // check if template and template histogram can be adpated in
        // TODO: user-defined checking interval
        float bc_templ = evolved_hist.match(templ_hist);
        float hu_templ_1 = evolved.match(templ);

        FourierDescriptor evolved_fd(evolved.mask, 64);
        float fd_templ = evolved_fd.match(templ_fd);

        hu1.push_back(hu_templ_1);
        fd1.push_back(fd_templ);

/* =============================================================================


        // detect successful tracking, so template can be adapted
//        if (bc_templ < 0.1) // just try out
//        {
//            evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
//            normalize(templ_hist);
//            frame.copyTo(templ_image);
//        }


        // TODO histogram adpation instead of replacement?
        //      consider more frames that keep this status

//        if ((cnt_frame % 19 == 0 && cnt_frame != 0) &&
//            (hu_templ < .15f && bc_templ < .35f))
//        {
//            evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
////            cv::normalize(templ_hist, templ_hist);
//            normalize(templ_hist);
////            frame.copyTo(templ_image);
//            cv::cvtColor(frame, templ_image, CV_GRAY2RGB);
//        }

//        bool lost = bc_templ > 0.2; // detected as lost
//        if (lost)
//        {
//            cv::rectangle(window_image, cv::Rect(5, 5, 20, 20), RED, -1);
//            evolved.contour_mask.copyTo(templ);
//            calc_hist(frame, templ_hist, templ);
//            normalize(templ_hist);
//            cv::cvtColor(frame, templ_image, CV_GRAY2RGB);
//        }

============================================================================= */

        // =====================================================================
        // = UPDATE PARTICLE FILTER                                            =
        // =====================================================================

        // update state and resampling
        pf.weighted_mean_estimate();
        pf.resample();
//        pf.resample_systematic();

        // =====================================================================
        // = DATA OUTPUT                                                       =
        // =====================================================================
        std::cout << boost::format("#%03d: bc-templ[%f] hu-templ-1[%f] fd-templ[%f]")
                     % cnt_frame % bc_templ % hu_templ_1 % fd_templ << std::endl;

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

    if (!matlab_file_path.empty())
    {
        std::ofstream m_output(matlab_file_path);
        if (m_output.is_open())
        {
            m_output << "hu1 = " << hu1 << ";" << std::endl;
            m_output << "fd1 = " << fd1 << ";" << std::endl;
            m_output.close();
        }
        else
            std::cerr << "Could not open: " << matlab_file_path << std::endl;
    }

    // =========================================================================
    // = VIDEO OUTPUT                                                          =
    // =========================================================================

    if (save_video)
    {
        videoOut.release();

        // convert to mp4 using avconv system call
        std::string name = output_path.substr(0, output_path.length()-4);
        std::stringstream ss;
        ss << "avconv -y -loglevel quiet -i " << output_path << " "
           << name + ".mp4";

        if (system(ss.str().c_str()) == -1)
        {
            std::cerr << "Error calling " << ss.str() << std::endl;
        }
        else
        {   // remove opencv created file
            ss.str("");
            ss << "rm " << output_path;
            if (system(ss.str().c_str()) == -1)
                std::cerr << "Could not remove: " << output_path << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
