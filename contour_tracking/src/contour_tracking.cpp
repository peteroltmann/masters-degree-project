#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

#include "Contour.h"
#include "RegBasedContours.h"
#include "ContourParticleFilter.h"

#define WINDOW_NAME "Image"

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
        sigma = 25.f;
    }

    int key = 0;
    cv::namedWindow(WINDOW_NAME);
    cv::Mat window_image;
    cv::Mat frame;

    RegBasedContours segm;
    ContourParticleFilter pf(num_particles);
    cv::Mat_<float> templ;
    fs2["templ"] >> templ;
    pf.init(templ);

    Contour templ_contour;
    templ.copyTo(templ_contour.contour_mask);

    // for this video: cv::cvtColor(frame, frame, CV_RGB2GRAY);
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

#ifdef PF_AC
    // first frame
    capture >> frame;

    segm.setFrame(frame);
    segm.init(templ);
    templ_contour.calc_energy(segm);
#else
    Contour contour;
    templ.copyTo(contour.contour_mask);
#endif

    int cnt_frame = 0;
    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;

        if (cvt_color)
            cv::cvtColor(frame, frame, CV_RGB2GRAY);

        cv::cvtColor(frame, window_image, CV_GRAY2BGR);

#ifdef PF_AC

        pf.predict();

        // calculate center of template contour (translation from [0, 0])
        cv::Moments m = cv::moments(templ, true);
        cv::Point2f center(m.m10/m.m00, m.m01/m.m00);
        cv::Mat_<float> templ_at_zero = (cv::Mat_<float>(4, 1) <<
                                      -center.x, -center.y, 0, 0);
        cv::Mat_<float> tmp = templ_at_zero + pf.state;

        // transformation and evolution: start from template
        templ.copyTo(pf.state_c.contour_mask);
        pf.state_c.transform_affine(tmp);
        pf.state_c.evolve_contour(segm, frame, num_iterations);
        pf.state_c.calc_energy(segm);

        // predcit deformations for each particle
        for (int i = 0; i < pf.num_particles; i++)
        {
            cv::Mat_<float> tmp = templ_at_zero + pf.p[i];

            templ.copyTo(pf.pc[i]->contour_mask);
            pf.pc[i]->transform_affine(tmp);
            pf.pc[i]->evolve_contour(segm, frame, num_iterations);
            pf.pc[i]->calc_energy(segm);
        }

        pf.calc_weight(templ_contour.energy, sigma);

        // draw particles
//        for (int i = 0; i < pf.num_particles; i++)
//        {
//            draw_contour(window_image, pf.pc[i]->contour_mask,
//                         cv::Scalar(0, 0, 255));
//        }

        // draw template and estimated contour
        // The "trick": additional evolution steps
//        pf.state_c.evolve_contour(segm, frame, 200);
        if (cnt_frame == 0)
            draw_contour(window_image, templ, cv::Scalar(255, 0, 0));
        draw_contour(window_image, pf.state_c.contour_mask,
                     cv::Scalar(255, 255, 255));

        pf.weighted_mean_estimate();
        pf.resample();
//        pf.resample_systematic();

        // adapt template to estimated state
//        pf.state_c.contour_mask.copyTo(templ_contour.contour_mask);
//        templ_contour.energy = pf.state_c.energy;

#else
        contour.evolve_contour(segm, frame, 50);
        draw_contour(window_image, contour.contour_mask,
                     cv::Scalar(255, 255, 255));
#endif // PF_AC

        cv::imshow(WINDOW_NAME, window_image);
        key = cv::waitKey(1);

        if (save_img_seq)
        {
            std::stringstream s;
            s << boost::format(save_img_path) % cnt_frame;
            cv::imwrite(s.str(), window_image);
        }

        if (save_video)
            videoOut << window_image;

        cnt_frame++;
    }
#ifdef SAVE_VIDEO
    videoOut.release();
#endif
    return EXIT_SUCCESS;
}
