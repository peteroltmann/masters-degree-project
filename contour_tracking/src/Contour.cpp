#include "Contour.h"
#include "StateParams.h"
#include "RegBasedContours.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

Contour::Contour() {}

Contour::~Contour() {}

void Contour::transform_affine(cv::Mat_<float>& state)
{
    // translate each point on the contour mask
    cv::Mat_<uchar> dst(contour_mask.size(), 0);
    for (int y = 0; y < contour_mask.rows; y++)
    {
        uchar* cm = contour_mask.ptr(y);
        for (int x = 0; x < contour_mask.cols; x++)
        {
            if (cm[x] == 1)
            {
                int xn = x + state(PARAM_X);
                int yn = y + state(PARAM_Y);
                if (yn >= 0 && yn < contour_mask.rows &&
                    xn >= 0 && xn < contour_mask.cols)
                {
                    dst(yn, xn) = 1;
                }
            }
        }
    }

    contour_mask = dst;

//    cv::imshow("ASD", contour_mask == 1);
//    cv::waitKey(0);
}


void Contour::evolve_contour(RegBasedContours& segm, cv::Mat& frame,
                             int iterations)
{
    segm.setFrame(frame);
    segm.init(contour_mask);

    // remember phi for weight calculation
    segm._phi.copyTo(phi_mu);

    for (int its = 0; its < iterations; its++)
    {
        segm.iterate();
    }

    // acutalize mask
    contour_mask.setTo(cv::Scalar(0));
    contour_mask.setTo(cv::Scalar(1), segm._phi <= 0);
}

void Contour::calc_energy(RegBasedContours &segm)
{
    segm.calcF();
    energy = cv::sum(cv::abs(segm._F))[0];
}
