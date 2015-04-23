#include "FourierDescriptor.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

FourierDescriptor::FourierDescriptor() {}

FourierDescriptor::FourierDescriptor(const cv::Mat_<uchar>& mask)
{
    init(mask);
}

FourierDescriptor::FourierDescriptor(const FourierDescriptor& other) :
    mask_size(other.mask_size),
    num_points(other.num_points),
    cp(other.cp)
{
    other.U.copyTo(U);
    other.Fc.copyTo(Fc);
    other.Fp.copyTo(Fp);
}

FourierDescriptor&FourierDescriptor::operator=(const FourierDescriptor& other)
{
    mask_size = other.mask_size;
    num_points = other.num_points;
    cp = other.cp;
    other.U.copyTo(U);
    other.Fc.copyTo(Fc);
    other.Fp.copyTo(Fp);

    return *this;
}

FourierDescriptor::~FourierDescriptor() {}

void FourierDescriptor::init(const cv::Mat_<uchar>& mask)
{
    U.release();
    mask_size = mask.size();

    // estimate outer contour(s)
    cv::Mat inOut = mask == 1;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    inOut.setTo(0);

    // take contour with most points
    if (contours.size() == 1)
        cp = contours[0];
    else
    {
        int max_points = 0, max_idx = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            if (max_points < contours[i].size())
            {
                max_points = contours[i].size();
                max_idx = i;
            }
        }
        cp = contours[max_idx];
    }

    num_points = cp.size();

    // create complex vector
    for (int i = 0; i < cp.size(); i++)
        U.push_back(cv::Vec2f(cp[i].x, cp[i].y));

    cv::dft(U, Fc); // apply DFT -> cartesian representation

    // calc polar representation
    cv::Mat_<float> planes[2];
    cv::split(Fc, planes);
    cv::cartToPolar(planes[0], planes[1], planes[0], planes[1]);
    cv::merge(planes, 2, Fp);

    // assure all frequencies != 0
    Fc += FLT_EPSILON;
    Fp += FLT_EPSILON;
/*
    // plot complex input values as points in order
    cv::Mat_<uchar> tmp = cv::Mat_<uchar>::zeros(mask.size());
    for (int i = 0; i < U.total(); i++)
    {
        cv::circle(tmp, cv::Point(U(i)[0], U(i)[1]), 0, 255);
        cv::imshow("Fourier input", tmp);
        cv::waitKey(10);
    }
*/
}

float FourierDescriptor::match(const FourierDescriptor& fd2)
{
    /*
     * 0...N --> low...high frequencies
     * translation invariance: F[0] := 0
     *     F[0] contains all translation related information
     * scale invariance: F[i] := F[i] / |F[1]|
     *     just scaling differences: F1[i] / |F2[1]|  =  F2[i] / F2[1]|)
     * rotation invariance: consider only |F[i]|
     *     rotation and starting point changes affect only the phase
     */

    // normalize using 2nd frequency (scaling information)
    cv::Mat_<cv::Vec2f> Fp1, Fp2; // note: also dividing phase, but not used
    cv::Mat_<cv::Vec2f> Fp1_tmp =     Fp /     Fp(    num_points-1)[0];
    cv::Mat_<cv::Vec2f> Fp2_tmp = fd2.Fp / fd2.Fp(fd2.num_points-1)[0];

    Fp1_tmp(0) = 0; // translation frequency
    Fp2_tmp(0) = 0;
    Fp1_tmp(    num_points-1) = 0; // scale frequency
    Fp2_tmp(fd2.num_points-1) = 0;

    // remove zero frequencies
    for (int i = 0; i < Fp1_tmp.total(); i++)
        if (Fp1_tmp(i) != cv::Vec2f(0, 0))
            Fp1.push_back(Fp1_tmp(i));

    for (int i = 0; i < Fp2_tmp.total(); i++)
        if (Fp2_tmp(i) != cv::Vec2f(0, 0))
            Fp2.push_back(Fp2_tmp(i));

    // assert equal non-zero frequencies
    if (Fp1.total() != Fp2.total())
    {
        std::cerr << "unequal amount of non-zero frequencies: " << Fp1.total()
                  << " (1) != " << Fp2.total() << " (2)" << std::endl;
        return -1;
    }

    // (Fp2 - Fp1) ^ 2
    cv::Mat_<cv::Vec2f> diff = Fp2 - Fp1;
    cv::pow(diff, 2, diff);

    return std::sqrt(cv::sum(diff)[0]); // only magnitude
}

cv::Mat_<uchar> FourierDescriptor::reconstruct()
{
    cv::Mat_<uchar> reconst_mask(mask_size, 0);
    cv::Mat_<cv::Vec2f> iU;

    cv::idft(Fc, iU, cv::DFT_SCALE); // inverse fourier transformation

    // create reconstructed contour mask
    std::vector<std::vector<cv::Point>> rp(1);
    for (int i = 0; i < iU.total(); i++)
        rp[0].push_back(cv::Point(std::round(iU(i)[0]), std::round(iU(i)[1])));

    cv::drawContours(reconst_mask, rp, 0, 1, CV_FILLED);
    return reconst_mask;
}

void FourierDescriptor::low_pass(int num_fourier)
{
    if (num_points < 2*num_fourier)
    {
        std::cerr << "Descriptor size ("<< num_points <<") are too small (< 2*"
                  << num_fourier << ")" << std::endl;
        return;
    }

    // set high frequencies to zero according to num_fourier
    int num_zero = num_points - 2*num_fourier;
    cv::Mat mask = cv::Mat::zeros(U.size(), CV_8U);
    cv::Mat roi(mask, cv::Rect(0, num_points/2 - num_zero/2, 1, num_zero));
    roi.setTo(1);
    Fc.setTo(0, mask);
    Fp.setTo(0, mask);
}
