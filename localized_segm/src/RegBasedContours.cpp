#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

RegBasedContours::RegBasedContours() {}

RegBasedContours::~RegBasedContours() {}

void RegBasedContours::apply(cv::Mat frame, cv::Mat initMask, cv::Mat& seg, int iterations,
                             float alpha)
{
    cv::Mat image;
    frame.copyTo(image);
    frame.convertTo(frame, CV_32F);

    // TODO assert frame.size() = initMask.size()

    // create a signed distance map (SDF) from mask
    cv::Mat phi = mask2phi(initMask);

    // main loop
    for (int its = 0; its < iterations; its++)
    {
        // "find curves narrow band"
        cv::Mat idx1 = phi <= 1.2f;
        cv::Mat idx2 = phi >= -1.2f;
        cv::Mat idx = idx1 == idx2;
        // TODO: no narrow band, just everything
        idx = cv::Mat::ones(phi.rows, phi.cols, CV_8U);

        // reinitialize segmenation image
        seg = cv::Mat::zeros(image.rows, image.cols, image.type());
        seg.setTo(255, phi < 0);

#ifdef SHOW_CONTOUR_EVOLUTION
        // show contours
        cv::Mat inOut;
        seg.copyTo(inOut);
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Mat out;
        image.copyTo(out);
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Image", out);
        cv::waitKey(1);
//        std::cout << "its: " << its << std::endl;
#endif

        // find interior and exterior mean
        cv::Mat intPts = phi <= 0;
        cv::Mat extPts = phi > 0;
        float meanInt = cv::mean(frame, intPts)[0];
        float meanExt = cv::mean(frame, extPts)[0];

        // (I(x)-u).^2-(I(x)-v).^2
        cv::Mat diffInt;
        cv::Mat diffExt;
        cv::Mat diffInt2;
        cv::Mat diffExt2;
        cv::Mat F;
        cv::subtract(frame, meanInt, diffInt, idx);
        cv::subtract(frame, meanExt, diffExt, idx);
        cv::multiply(diffInt, diffInt, diffInt2);
        cv::multiply(diffExt, diffExt, diffExt2);
        cv::subtract(diffInt2, diffExt2, F, idx);

        // dphidt = F./max(abs(F)) + alpha*curvature;
        // % gradient descent to minimize energy
        cv::Mat dphidt;
        double maxF;
        cv::minMaxIdx(cv::abs(F), NULL, &maxF, NULL, NULL, idx);
        dphidt = F * (1.f/((float) maxF));
//        cv::Mat curvature;
//        calcCurvature(phi, curvature);
//        curvature = curvature * 0.2f;
//        dphidt = dphidt + curvature;

        // % maintain the CFL condition
        // dt = .45/(max(dphidt)+eps);
        double maxdphidt;
        cv::minMaxIdx(dphidt, NULL, &maxdphidt, NULL, NULL, idx);
        float dt = .45f / ((float) maxdphidt + FLT_EPSILON); // TODO ???
//        dt = .5f;

        cv::Mat dt_dphidt = dt * dphidt;
        cv::add(phi, dt_dphidt, phi, idx);
    }
}

void RegBasedContours::calcCurvature(cv::Mat phi, cv::Mat curvature,
                                     cv::Mat mask)
{
    if (curvature.empty() || curvature.size() != phi.size() ||
        curvature.type() != phi.type())
    {
        curvature.create(phi.size(), phi.type());
    }

    for (int i = 1; i < phi.rows-1; i++)
    {
        for (int j = 1; j < phi.cols-1; j++)
        {
            float phixx = (phi.at<float>(i+1,j) - phi.at<float>(i,j))
                        - (phi.at<float>(i,j) - phi.at<float>(i-1,j));
            float phiyy = (phi.at<float>(i,j+1) - phi.at<float>(i,j))
                        - (phi.at<float>(i,j) - phi.at<float>(i,j-1));
            float phixy = (phi.at<float>(i+1,j+1) - phi.at<float>(i-1,j+1))
                        - (phi.at<float>(i+1,j-1) - phi.at<float>(i-1,j-1));
            phixy *= 1.f/4.f;
            float phix = (phi.at<float>(i+1,j) - phi.at<float>(i-1,j));
            phix *= 1.f/.2f;
            float phiy = (phi.at<float>(i,j+1) - phi.at<float>(i, j-1));

            curvature.at<float>(i,j) = (phixx*phiy*phiy - 2.f*phiy*phix*phixy + phiyy*phix*phix) / std::pow((phix*phix + phiy*phiy + FLT_EPSILON), 3.f/2.f);
        }
    }
}

cv::Mat RegBasedContours::mask2phi(cv::Mat mask)
{
    // TODO: understand
    // phi=bwdist(init_a)-bwdist(1-init_a)+im2double(init_a)-.5;
    cv::Mat phi(mask.rows, mask.cols, CV_32F);
    cv::Mat dist1, dist2;
    cv::Mat maskf;
    mask.convertTo(maskf, CV_32F);

    // Note:
    // matlab: distance to nearest NON-ZERO pixel
    // opencv: distance to nearest ZERO pixel
    // --> swap of mask and (1-mask)
    cv::distanceTransform(cv::Scalar::all(1) - mask, dist1, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    cv::distanceTransform(mask, dist2, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    phi = dist1 - dist2 + maskf - cv::Scalar::all(0.5f);
    return phi;
}
