#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <iostream>

#include "Random.h"
#include "PosParticleFilter.h"

#define WINDOW_NAME "Image Position Tracking"

int main(int argc, char** argv)
{
    cv::Size img_size(340, 360);
    cv::Mat frame; // simulated source iamge
    cv::Mat windowImg; // draw and display
    cv::Size templ_size = cv::Size(50, 50); // template size (white rectangle)
    cv::Rect bounds(0, 0, img_size.width, img_size.height);

    int T = 50;
    int pos_x = 50, pos_y = 50;

    PosParticleFilter pf(200);
    pf.init(50, 50);

    for (int t = 0; t < T; t++)
    {

        int x = pos_x - templ_size.width / 2;
        int y = pos_y - templ_size.height / 2;
        int width = templ_size.width;
        int height = templ_size.height;

        frame = cv::Mat::zeros(img_size, CV_8U);
        cv::Mat frame_roi = frame(cv::Rect(x, y, width, height) & bounds);
        frame_roi.setTo(255);


        pf.predict();
        pf.calcWeight(frame, templ_size);
        pf.weightedMeanEstimate();

        cv::Point state_point(round(pf.state(PARAM_X)), round(pf.state(PARAM_Y)));
        if(!bounds.contains(state_point))
            pf.redistribute(frame.size());


        std::cout << "Estimate: ";
        std::cout << boost::format("[%3d, %3d]")
                     % std::round(pf.state(PARAM_X))
                     % std::round(pf.state(PARAM_Y))
                  << " - " << pf.confidence << std::endl;
        std::cout << "Real Pos: ";
        std::cout << boost::format("[%3d, %3d]") % pos_x % pos_y << std::endl;

        windowImg = cv::Mat::zeros(frame.size(), CV_8UC3); // reset display
        windowImg.setTo(cv::Scalar(255, 255, 255), frame == 255);

        // draw particles
//        for (int i = 0; i < pf.num_p; i++)
//        {
//            x = std::round(pf.p[i](PARAM_X)) - width / 2;
//            y = std::round(pf.p[i](PARAM_Y)) - height / 2;

//            cv::Rect region = cv::Rect(x, y, width, height) & bounds;
//            cv::rectangle(windowImg, region, cv::Scalar(0, 0, 255));
//        }

        // draw estimate
        x = std::round(pf.state(PARAM_X)) - width / 2;
        y = std::round(pf.state(PARAM_Y)) - height / 2;

        cv::Rect region = cv::Rect(x, y, width, height) & bounds;
        cv::rectangle(windowImg, region, cv::Scalar(255, 0, 0));


        cv::imshow(WINDOW_NAME, windowImg);
        cv::waitKey(250);

        pf.resample();
//        pf.resampleSystematic();

        pos_x += Random::getRNG().uniform(-1, 5);
        pos_y += Random::getRNG().uniform(-1, 5);
    }
    return EXIT_SUCCESS;
}
