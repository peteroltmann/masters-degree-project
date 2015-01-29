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
#define WINDOW_FRAME_NAME "Frame"
#define WINDOW_TEMPALTE_NAME "Template"
#define WINDOW_RECONSTR_NAME "Reconstructed"
#define WINDOW_RECONSTR_TEMPL_NAME "Reconstructed Template"

#define TEXT_POS cv::Point(10, 20)

int main(int argc, char *argv[])
{
    // =========================================================================
    // = PARAMETERIZATION                                                      =
    // =========================================================================

    // see 'parameterization.yml' for description
    int num_particles;
    int num_iterations;
    float sigma;
    int num_fourier;
    float fd_threshold;
    int num_free_frames;
    std::string templ_path;
    bool cvt_color; // TODO
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
    fs["num_fourier"] >> num_fourier;
    fs["fd_threshold"] >> fd_threshold;
    fs["num_free_frames"] >> num_free_frames;
    fs["templ_path"] >> templ_path;
    fs["cvt_color"] >> cvt_color;
    fs["input_path"] >> input_path;
    fs["save_video"] >> save_video;
    fs["fps"] >> fps;
    fs["output_path"] >> output_path;
    fs["save_img_seq"] >> save_img_seq;
    fs["save_img_path"] >> save_img_path;
    fs["matlab_file_path"]  >> matlab_file_path;

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
    if (num_fourier < 1)
    {
        std::cout << "invalid value for 'num_fourier', '10' used instead"
                  << std::endl;
        num_fourier = 10;
    }

    if (fd_threshold <= 0.f)
    {
        std::cout << "invalid value for 'fd_threshold', '0.01' used instead"
                  << std::endl;
        fd_threshold = .01f;
    }
    if (num_free_frames <= 0)
    {
        std::cout << "invalid value for 'num_free_frames', '10' used instead"
                  << std::endl;
        num_free_frames = 10;
    }

    // =========================================================================
    // = DECLARATION AND INITIALIZATION                                        =
    // =========================================================================

    // windows
    int key = 0;
    cv::namedWindow(WINDOW_NAME);

    // images
    cv::Mat frame; // current frame
    cv::Mat frame_gray; // current frame (grayscale)
    cv::Mat templ_image; // image of the template
    cv::Mat window_frame; // current frame for drawing output
    cv::Mat window_templ; // current template frame for drawing output

    // template contour, histogram and fourier descriptor
    cv::Mat_<uchar> in;
    fs2["templ"] >> in;
    Contour templ(in);
    Contour templ_frame0(in);
    Histogram templ_hist;
    FourierDescriptor templ_fd(templ.mask);
    templ_fd.low_pass(num_fourier);

    cv::Mat_<float> hu1; // hu values for matlab output
    cv::Mat_<float> fd1; // fd values for matlab output

    Contour templ_next;
    cv::Mat templ_image_next;
    int next_free = 0;
    int last_occluded = num_free_frames;

    RegBasedContours segm; // object providing the contour evolution algorithm

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

    // =========================================================================
    // = MAIN LOOP                                                             =
    // =========================================================================

    int cnt_frame = 0;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty()) // end of image sequence
            break;

        cv::cvtColor(frame, frame_gray, CV_RGB2GRAY);
        frame.copyTo(window_frame);

        cv::Rect bounds(0, 0, frame.cols, frame.rows); // outer frame bounds

        // calc template histogram on first frame
        if (templ_hist.empty())
        {
            // calc template histogram
            cv::Mat frame_roi(frame, templ.bound);
            templ_hist.calc_hist(frame_roi, RGB, templ.roi);

            // save frame as template image
            frame.copyTo(templ_image);
        }

        templ_image.copyTo(window_templ);

        // =====================================================================
        // = TRYOUT                                                            =
        // =====================================================================
/*
        FourierDescriptor fd(templ.mask);

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
        std::cout << "M" << M << std::endl;

//        cv::Mat M = (cv::Mat_<float>(2, 3) <<  1, 0, -50,
//                                               0, 1, -50);

        cv::imshow("Reconstruct 1", templ.mask == 1);
        cv::waitKey(1);
        cv::warpAffine(templ.mask, templ.mask, M, templ.mask.size());
//        templ.mask.setTo(0);
//        cv::rectangle(templ.mask, cv::Rect(100, 100, 100, 100), 1, -1);
        cv::imshow("Reconstruct 2", templ.mask == 1);
        cv::waitKey();

        FourierDescriptor fd2(templ.mask);
        fd.low_pass(num_fourier);
        fd2.low_pass(num_fourier);

        float match_fd = fd.match(fd2);
        std::cout << "match_fd: " << match_fd << std::endl;

        // reconstruct
        cv::Mat reconst_mask = fd.reconstruct();
        cv::imshow("Reconstruct 1", reconst_mask == 1);

        cv::Mat reconst2_mask = fd2.reconstruct();
        cv::imshow("Reconstruct 2", reconst2_mask == 1);
        cv::waitKey();

        return EXIT_SUCCESS; // ################################################
*/
        // =====================================================================
        // = PARTICLE FILTER                                                   =
        // =====================================================================

        pf.predict();

        // check if target is lost (out of bounds)
        if (!bounds.contains(cv::Point(std::round(pf.state(PARAM_X)),
                                       std::round(pf.state(PARAM_Y)))))
        {
            pf.redistribute(frame.size());
            continue;
        }

        pf.calc_weight(frame, templ.bound.size(), templ_hist, sigma);

/*        // print weights
        for (int i = 0; i < pf.num_particles; i++)
        {
            std::cout << pf.w[i] << std::endl;
        }
*/

/*        // draw particles
        for (int i = 0; i < pf.num_particles; i++)
        {
            cv::Rect pi_rect = pf.state_rect(templ.bound.size(), bounds, i);
            cv::rectangle(window_frame, pi_rect, RED, 1);
        }
*/

        // get predicted estimate rectangle
        cv::Rect estimate_rect = pf.state_rect(templ.bound.size(), bounds);

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

        // =====================================================================
        // = AFTER CONTOUR EVOLUTION                                           =
        // =====================================================================

        // calc evolved contour histogram
        Histogram evolved_hist;
        cv::Mat frame_roi(frame, evolved.bound);
        evolved_hist.calc_hist(frame_roi, RGB, evolved.roi);

        // create evolved contour fourier descriptor
        FourierDescriptor evolved_fd(evolved.mask);
        evolved_fd.low_pass(num_fourier);

        // calculate matching values between evolved contour and template
        float bc_templ = evolved_hist.match(templ_hist);
        float hu_templ = evolved.match(templ);
        float fd_templ = evolved_fd.match(templ_fd);

        hu1.push_back(hu_templ);
        fd1.push_back(fd_templ);

        // replacement contour if object is lost after evolution or occluded
        Contour evolved_repl;

        // =====================================================================
        // = HISTOGRAMM ADPATION HANDLING                                      =
        // =====================================================================

/*
        if (bc_templ < bc_threshold)
        {
            // object contour found
            // occlusion or deformation possible (but not considerable)
            // adapt histogram
        }
        else // if (bc_templ > bc_threshold_repl)
        {
            // lost object after evolution
            // use evolved_repl
        }
*/

//        if (bc_templ >= 0.4)

        // =====================================================================
        // = OCCLUSION HANDLING                                                =
        // =====================================================================

/*
        if (fd_templ < fd_threshold)
        {
            // apdapt (replace) template
            // either: check last x frames
            // or: add as additional template (from now: match all templates)
        }
        else // if (fd_templ > fd_threshold_repl)
        {
            // occluded or deformed
            // check occlusion: characteristic views
            // use evolved_repl
        }
*/

        // check for occlusion
        if (fd_templ >= fd_threshold) // 0.01: fish, 0.05: plane
        {
            // reset occlusion counters
            last_occluded = num_free_frames;
            next_free = 0;

            // set (reconstructed) template as replacement contour
//            evolved.set_mask(templ.mask);
            evolved_repl.set_mask(templ_fd.reconstruct());
            evolved_repl.transform_affine(pf.state);
        }
        else // not occluded
            last_occluded <= 0 ? 0 : last_occluded--;

        // handle template replacement
        if (!last_occluded)
        {
            // last x frames: free
            if (next_free == 0)
            {
                // template good enough for replacement
                if (fd_templ < fd_threshold/2.f)
                {
                    // set possible next template
                    templ_next.set_mask(evolved.mask);
                    frame.copyTo(templ_image_next);
                    next_free++;
                }
                else
                    next_free = 0; // try next (if still not occluded)
            }
            // next x frames: free
            else if (next_free == num_free_frames-1)
            {
                // replace template
                templ.set_mask(templ_next.mask);
                templ_fd.init(templ_next.mask);
                templ_fd.low_pass(num_fourier);

                // calc template histogram
//                cv::Mat frame_roi(templ_image_next, templ.bound);
//                templ_hist.calc_hist(frame_roi, RGB, templ.roi);

                // save frame as template image
                templ_image_next.copyTo(templ_image);
                templ_image_next.copyTo(window_templ);

                next_free = 0; // reset not occluded (next) counter
            }
            else
                next_free++;
        }

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

        std::cout << boost::format("#%03d: bc-templ[ %f ] hu-templ[ %f ] "
                                   "fd-templ[ %f ] last-occluded[ %d ] "
                                   "next-free[ %d ]")
                     % cnt_frame % bc_templ % hu_templ % fd_templ
                     % last_occluded % next_free << std::endl;

        // =====================================================================
        // = IMAGE OUTPUT                                                      =
        // =====================================================================

        // draw predicted estimate
        cv::rectangle(window_frame, estimate_rect, GREEN, 1);

        // draw contours
        templ.draw(window_templ, BLUE);
        evolved.draw(window_frame, WHITE);
//        cv::rectangle(window_templ, templ.bound, BLUE);
//        cv::rectangle(window_frame, evolved.bound, WHITE);

        if (!evolved_repl.empty())
            evolved_repl.draw(window_frame, BLUE);

        // vieo output
        cv::Mat video_frame;
        window_frame.copyTo(video_frame);

        // put text
        int font = CV_FONT_HERSHEY_SIMPLEX;

        // reconstructed images - bottom
        cv::Mat er = evolved_fd.reconstruct() == 1;
        cv::Mat tr = templ_fd.reconstruct() == 1;
        cv::putText(er, WINDOW_RECONSTR_NAME, TEXT_POS, font, .4, WHITE);
        cv::putText(tr, WINDOW_RECONSTR_TEMPL_NAME, TEXT_POS, font, .4, WHITE);

        // frame (particle filter, evolved) - template
        cv::putText(window_frame, WINDOW_FRAME_NAME, TEXT_POS, font, .4, WHITE);
        cv::putText(window_templ, WINDOW_TEMPALTE_NAME, TEXT_POS, font, .4, WHITE);

        // concatenate
        cv::Mat bottom, top;
        cv::hconcat(er, tr, bottom);
        cv::cvtColor(bottom, bottom, CV_GRAY2RGB);
        cv::hconcat(window_frame, window_templ, top);
        cv::vconcat(top, bottom, top);

        cv::imshow(WINDOW_NAME, top);
        key = cv::waitKey(1);

        if (save_img_seq)
        {
            std::stringstream s;
            s << boost::format(save_img_path) % cnt_frame;
            cv::imwrite(s.str(), window_frame);
        }

        if (save_video)
            videoOut << video_frame;

        // pause on space
        if (key == ' ')
        {
            key = cv::waitKey(0);
            while (key != ' ' && key != 'q')
                key = cv::waitKey(0);
        }

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
            std::cerr << "Error calling " << ss.str() << std::endl;
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
