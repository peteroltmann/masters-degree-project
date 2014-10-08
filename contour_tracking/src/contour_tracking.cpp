#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

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
    cv::Mat window_image;
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

#ifdef PF_AC
    // first frame
    capture >> frame;

    segm.setFrame(frame);
    segm.init(templ);
//    segm.iterate();
    templ_contour.calc_energy(segm);
    std::cout << templ_contour.energy << std::endl;
#else
    Contour contour;
    templ.copyTo(contour.contour_mask);
#endif

    while (key != 'q')
    {
        capture >> frame;
        if (frame.empty())
            break;
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

        templ.copyTo(pf.state_c.contour_mask);
        pf.state_c.transform_affine(tmp);
        pf.state_c.evolve_contour(segm, frame, NUM_ITERATIONS);
        pf.state_c.calc_energy(segm);

        // DO TRANSFORM AND EVOLUTION: START FROM TEMPLATE
        // TODO: UPDATE EACH CONTOUR WITH EVOLUTION PREDICTION
        for (int i = 0; i < pf.num_particles; i++)
        {
            cv::Mat_<float> tmp = templ_at_zero + pf.p[i];

            templ.copyTo(pf.pc[i]->contour_mask);
            pf.pc[i]->transform_affine(tmp);
            pf.pc[i]->evolve_contour(segm, frame, NUM_ITERATIONS);
            pf.pc[i]->calc_energy(segm);
        }

        pf.calc_weight(templ_contour.energy);

//        float w_max = 0.f;
//        int w_max_idx = 0;
//        for (int i = 0; i < pf.num_particles; i++)
//        {
//            if (pf.w[i] > w_max)
//            {
//                w_max = pf.w[i];
//                w_max_idx = i;
//            }
//        }


        // draw particles
//        for (int i = 0; i < pf.num_particles; i++)
//        {
//            draw_contour(window_image, pf.pc[i]->contour_mask,
//                         cv::Scalar(0, 0, 255));
//        }

        // draw template and estimated contour
        // The "trick": additional evolution steps
//        pf.state_c.evolve_contour(segm, frame, 200);
        draw_contour(window_image, templ, cv::Scalar(255, 0, 0));
        draw_contour(window_image, pf.state_c.contour_mask,
                     cv::Scalar(255, 255, 255));
//        draw_contour(window_image, pf.pc[w_max_idx]->contour_mask,
//                     cv::Scalar(0, 255, 0));

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

#ifdef SAVE_VIDEO
        videoOut << window_image;
#endif

    }
#ifdef SAVE_VIDEO
    videoOut.release();
#endif
    return EXIT_SUCCESS;
}
