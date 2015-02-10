#include "ObjectTracker.h"

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
#include "Selector.h"
#include "Constants.h"

#define TEXT_POS cv::Point(10, 20)

ObjectTracker::ObjectTracker() {}

ObjectTracker::~ObjectTracker() {}

int ObjectTracker::run(std::string param_path)
{
    // =========================================================================
    // = PARAMETERIZATION                                                      =
    // =========================================================================

    // see 'parameterization.yml' for description
    int num_particles;
    int num_iterations;
    float sigma;
    float bc_threshold;
    float bc_threshold_adapt;
    float a;
    int num_fourier;
    float fd_threshold;
    bool select_start_rect;
    cv::Rect start_rect;
    std::string input_path;
    std::vector<std::string> char_views;
    std::string output_path;
    double fps;
    std::string save_img_path;
    std::string matlab_file_path;
    int method;
    bool localized;
    int rad;
    float alpha;

    cv::FileStorage fs(param_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error opening '" << param_path << "'" << std::endl;
        std::cerr << "Specify parameterization file as argument or use "
                     "default: '../parameterization.yml'" << std::endl;
        return EXIT_FAILURE;
    }

    fs["num_particles"] >> num_particles;
    fs["num_iterations"] >> num_iterations;
    fs["sigma"] >> sigma;
    fs["bc_threshold"] >> bc_threshold;
    fs["bc_threshold_adapt"] >> bc_threshold_adapt;
    fs["a"] >> a;
    fs["num_fourier"] >> num_fourier;
    fs["fd_threshold"] >> fd_threshold;
    fs["select_start_rect"] >> select_start_rect;
    fs["start_rect"] >> start_rect;
    fs["input_path"] >> input_path;
    fs["char_views"] >> char_views;
    fs["output_path"] >> output_path;
    fs["fps"] >> fps;
    fs["save_img_path"] >> save_img_path;
    fs["matlab_file_path"]  >> matlab_file_path;
    fs["method"] >> method;
    fs["localized"] >> localized;
    fs["rad"] >> rad;
    fs["alpha"] >> alpha;

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
    if (bc_threshold <= 0.f || bc_threshold >= 1.f)
    {
        std::cout << "invalid value for 'bc_threshold', '0.25' used instead"
                  << std::endl;
        bc_threshold = .25f;
    }
    if (bc_threshold_adapt < 0.f || bc_threshold_adapt > 1.f)
    {
        std::cout << "invalid value for 'bc_threshold_adapt', '0' used instead"
                  << std::endl;
        bc_threshold_adapt = 0.f;
    }
    if (a <= 0.f || a > 1.f)
    {
        a = .1f;
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

    // select_start_rect: bool, nothing to check

    if (!select_start_rect && (
         start_rect.width <= 0 || start_rect.height <= 0 ||
         start_rect.x     <  0 || start_rect.y      <  0 ))
    {
        std::cerr << "Invalid size for starting rectangle: "
                  << start_rect << std::endl;
        return EXIT_FAILURE;
    }

    // input_path: on open VideoCapture
    // char_views: on first frame evolution: if empty, only use first frame
    // output_path: on open VideoWriter
    // fps: on open VideoWriter
    // save_img_path: on saving image sequence
    // matlab_file_path: on matlab output

    if (method != CHAN_VESE && method != YEZZI)
    {
        std::cerr << "Invalid method: " << method << std::endl;
        return EXIT_FAILURE;
    }

    // localized: bool, nothing to check
    if (rad <= 0)
        rad = 18;
    if (alpha <= 0 || alpha >= 1)
        alpha = .2f;

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
    Contour templ;
    Histogram templ_hist;
    FourierDescriptor templ_fd;

    cv::Mat_<float> hu1; // hu values for matlab output
    cv::Mat_<float> fd1; // fd values for matlab output

    // characteristic views
    int last_match_idx = 0;
    std::vector<FourierDescriptor> char_views_fd(char_views.size() + 1);

    // object providing the contour evolution algorithm
    RegBasedContours segm(Method(method), localized, rad, alpha);

    // init particle filter
    ParticleFilter pf(num_particles);

    // open input image sequence or video
    cv::VideoCapture capture(input_path);
    if (!capture.isOpened())
    {
        std::cerr << "Failed to open capture: '" << input_path << "'"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // create video output, if ouput path is set
    cv::VideoWriter video_out;
    cv::VideoWriter video_out_details;
    if(!output_path.empty())
    {
        std::string name = output_path.substr(0, output_path.length()-4);
        std::string ext = output_path.substr(output_path.length()-4);

        if (ext != ".avi")
            throw cv::Exception(-1, "output_path must end with '.avi'",
                                "ObjectTracker::run()", "ObjectTracker.cpp", 0);

        if (fps <= 0)
        {
            double input_fps = capture.get(CV_CAP_PROP_FPS);
            fps = input_fps;
        }

        // get frame size from video input
        cv::Size frame_size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
                            capture.get(CV_CAP_PROP_FRAME_HEIGHT));

        video_out.open(output_path, CV_FOURCC('X', 'V', 'I', 'D'), fps,
                      frame_size, true);
        if (!video_out.isOpened())
        {
            std::cerr << "Could not write output video" << std::endl;
            return EXIT_FAILURE;
        }

        video_out_details.open(name + "_details.avi",
                               CV_FOURCC('X', 'V', 'I', 'D'), fps,
                               cv::Size(frame_size.width*2,
                                        frame_size.height*2), true);
        if (!video_out_details.isOpened())
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

        // =====================================================================
        // = FIRST FRAME                                                       =
        // =====================================================================

        // evolve contour on first frame from starting rectangle
        // to determine the template contour
        if (templ.empty())
        {
            // select starting rectangle
            if (select_start_rect)
            {
                // displace frame for selection
                std::cout << "Please select a starting rectangle in the frame. "
                             "Then press any key to start." << std::endl;

                Selector selector(WINDOW_NAME, frame); // handle mouse callback
                cv::imshow(WINDOW_NAME, frame);
                while (!selector.is_valid())
                {
                    cv::waitKey();
                    if (!selector.is_valid())
                    {
                        std::cerr << "Invalid selection: "
                                  << selector.get_selection() << std::endl;
                    }
                }
                start_rect = selector.get_selection();

            }

            // check start_rect size
            if (start_rect.x + start_rect.width  > frame.cols ||
                start_rect.y + start_rect.height > frame.rows)
            {
                std::cerr << "Error: starting rectangle " << start_rect
                          << " does not fit with frame size " << frame.size()
                          << std::endl;
                return EXIT_FAILURE;
            }

            std::cout << boost::format("Starting with rectangle: "
                                       "[ %3d, %3d, %3d, %3d ]")
                         % start_rect.x % start_rect.y % start_rect.width
                         % start_rect.height << std::endl;

            // create mask and evolve contour
            cv::Mat start_mask = cv::Mat::zeros(frame.size(), CV_8U);
            cv::Mat roi(start_mask, start_rect);
            roi.setTo(1);
            templ.set_mask(start_mask);
            templ.evolve(segm, frame_gray, num_iterations);

            // init fourier descriptor with template contour
            templ_fd.init(templ.mask);
            templ_fd.low_pass(num_fourier);

            // init characteristic views, first: template contour
            cv::Mat_<uchar> view_mask;
            char_views_fd[0].init(templ.mask);
            char_views_fd[0].low_pass(num_fourier);
            for (int i = 0; i < char_views.size(); i++)
            {
                view_mask = cv::imread(char_views[i], 0);
                if (view_mask.empty())
                {
                    std::cerr << "Error: can not open characteristic view: '"
                              << char_views[i] << "'" << std::endl;
                    return EXIT_FAILURE;
                }
                view_mask.setTo(1, view_mask);

                char_views_fd[i+1].init(view_mask);
                char_views_fd[i+1].low_pass(num_fourier);
            }

            // calc template histogram
            cv::Mat frame_roi(frame, templ.bound);
            templ_hist.calc_hist(frame_roi, RGB, templ.roi);

            // save frame as template image
            frame.copyTo(templ_image);

            // init particle filter
            pf.init(templ.bound);
        }

        // reset window template to template image every frame
        templ_image.copyTo(window_templ);

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

        // matching char_views
        std::vector<float> fd_char_views(char_views_fd.size());
        for (int i = 0; i < char_views_fd.size(); i++)
            fd_char_views[i] = evolved_fd.match(char_views_fd[i]);

        if (!matlab_file_path.empty())
        {
            hu1.push_back(hu_templ);
            fd1.push_back(fd_templ);
        }

        // replacement contour if object is lost after evolution or occluded
        Contour evolved_repl;

        // =====================================================================
        // = HISTOGRAMM ADPATION HANDLING                                      =
        // =====================================================================

        // object contour found
        // occlusion or deformation possible (but not considerable)
        // --> adapt histogram
        if (bc_templ < bc_threshold_adapt)
            templ_hist.adapt(evolved_hist, a);

        if (bc_templ >= bc_threshold) // lost object after contour evolution
        {
            // create replacement "contour": use PF estimate
            cv::Mat_<uchar> tmp(frame.size(), 0);
            cv::Mat_<uchar> roi(tmp, estimate_rect);
            roi.setTo(1);
            evolved_repl.set_mask(tmp);
        }

        // =====================================================================
        // = OCCLUSION HANDLING                                                =
        // =====================================================================

        else
        {
            // estimate best matching view
            int match_idx = -1;
            float match_min = fd_threshold;
            for (int i = 0; i < fd_char_views.size(); i++)
            {
                if (fd_char_views[i] < fd_threshold &&
                    fd_char_views[i] < match_min)
                {
                    match_idx = i;
                    match_min = fd_char_views[i];
                }
            }

            if (match_idx == -1) // occlusion / unknown deformation
            {
                // create replacement contour

                // reconstruct from last matching fourier descriptor
                evolved_repl.set_mask(char_views_fd[last_match_idx].
                                      reconstruct());

                // resize repl roi to evolved bound
                float ar   = 0.f;
                float len1 = evolved.bound.width; // (len2 * ar = len1)
                float len2 = evolved_repl.bound.width;
                if (evolved.bound.height < len1) // use height as scaling base
                {
                    len1 = evolved.bound.height;
                    len2 = evolved_repl.bound.height;
                }
                ar = len1 / len2;

                float width = evolved_repl.bound.width * ar;
                float height = evolved_repl.bound.height * ar;

                // assure new size to by <= frame size
                if (width > frame.cols)
                    width = frame.cols;
                if (height > frame.rows)
                    height = frame.rows;

                cv::Mat tmp;
                cv::resize(evolved_repl.roi, tmp, cv::Size(width, height));

                // zero-pad to frame size
                int bottom = (frame.rows - tmp.rows);
                int right  = (frame.cols - tmp.cols);
                cv::copyMakeBorder(tmp, tmp, 0, bottom, 0, right,
                                   cv::BORDER_CONSTANT, 0);

                // center of evolved contour = center of replacement contour
                cv::Mat_<float> s(NUM_PARAMS, 1);
                cv::Moments m = cv::moments(evolved.mask, true);
                cv::Point2f center(m.m10/m.m00, m.m01/m.m00);
                s(PARAM_X) = center.x;
                s(PARAM_Y) = center.y;

                evolved_repl.set_mask(tmp);
                evolved_repl.transform_affine(s);
            }
            else
                last_match_idx = match_idx; // for replacement contour
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
                                   /*"fd-templ[ %f ] last-occluded[ %d ] "
                                   "next-free[ %d ]"*/)
                     % cnt_frame % bc_templ % hu_templ/* % fd_templ
                     % last_occluded % next_free << std::endl*/;

        std::cout << "fd_char_views[ ";
        for (int i = 0; i < fd_char_views.size(); i++)
            std::cout << boost::format("%1.4f") % fd_char_views[i] << " ";
        std::cout << "]" << std::endl;

        // =====================================================================
        // = IMAGE OUTPUT                                                      =
        // =====================================================================
/*
        if (cnt_frame >= 280 && cnt_frame <= 300)
        {
            std::stringstream s;
            s << boost::format("/home/peter/Desktop/cv_fish/cv_fish_%03d.png")
                 % cnt_frame;
//            cv::imwrite(s.str(), evolved_fd.reconstruct() == 1);
            cv::imwrite(s.str(), evolved.mask == 1);
//            cv::waitKey();
        }
*/
        // draw predicted estimate
        // cv::rectangle(window_frame, estimate_rect, GREEN, 1);

        // draw contours
        templ.draw(window_templ, BLUE);
        // cv::rectangle(window_templ, templ.bound, BLUE);
        // cv::rectangle(window_frame, evolved.bound, WHITE);

        if (!evolved_repl.empty())
            evolved_repl.draw(window_frame, WHITE);
        else
            evolved.draw(window_frame, WHITE);

        // vieo output
        cv::Mat video_frame;
        window_frame.copyTo(video_frame);

        // draw predicted estimate
        cv::rectangle(window_frame, estimate_rect, GREEN, 1);

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

        // save image, if path is set
        if (!save_img_path.empty())
        {
            try
            {
                std::stringstream s;
                s << boost::format(save_img_path) % cnt_frame;
                if (!cv::imwrite(s.str(), window_frame))
                {
                    std::cerr << "Error: could not write image: '" << s.str()
                              << "'" << std::endl;
                }
            }
            catch (cv::Exception const&) {} // OpenCV prints error message
            catch (std::exception const& e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }

        if (video_out.isOpened())
            video_out << video_frame;
        if (video_out_details.isOpened())
            video_out_details << top;

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
    // = MATLAB OUTPUT                                                         =
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

    if (video_out.isOpened())
        video_out.release();
    if (video_out_details.isOpened())
        video_out_details.release();
}
