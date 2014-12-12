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
    // TODO find contour to get rid of blobs
    contour_mask.setTo(cv::Scalar(0));
    contour_mask.setTo(cv::Scalar(1), segm._phi <= 0);
}

void Contour::calc_energy(cv::Mat& frame)
{

    // calc energy
    float sumInt = 0, sumExt = 0;
    float cntInt = 0, cntExt = 0;
    float meanInt, meanExt;

    for (int y = 0; y < contour_mask.rows; y++)
    {
        uchar* I = frame.ptr<uchar>(y);
        uchar* cm = contour_mask.ptr<uchar>(y);
        for (int x = 0; x < contour_mask.cols; x++)
        {
            if (cm[x] == 1)
            {
                sumInt += I[x];
                cntInt++;
            }
            else
            {
                sumExt += I[x];
                cntExt++;
            }
        }
    }
    meanInt = sumInt / cntInt;
    meanExt = sumExt / cntExt;

    float E = 0;
    for (int y = 0; y < contour_mask.rows; y++)
    {
        uchar* I = frame.ptr<uchar>(y);
        uchar* cm = contour_mask.ptr<uchar>(y);
        for (int x = 0; x < contour_mask.cols; x++)
        {
            if (cm[x] == 1)
                E += std::pow(I[x] - meanInt, 2) / cntInt;
            else
                E += std::pow(I[x] - meanExt, 2) / cntExt;
        }
    }

    energy = E;
}

void Contour::calc_distance(RegBasedContours& segm)
{
    float hs = 0.f, hi = 0.f, h = 0.f;
    distance = 0.f;
    for (int y = 0; y < phi_mu.rows; y++)
    {
        const float* phi = segm._phi.ptr<float>(y);
        const float* phiMu = phi_mu.ptr<float>(y);
        for (int x = 0; x < phi_mu.cols; x++)
        {
            if (phi[x] >= 0)
                hs = 1.f;
            if (phiMu[x] >= 0)
                hi = 1.f;
            h = (hs + hi) / 2.f;

            distance += std::pow(phiMu[x] - phi[x], 2) * h;
        }
    }
}
