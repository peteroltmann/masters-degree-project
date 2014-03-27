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
#ifdef DEBUG
        clock_t clock1, clock2;
        clock1 = clock();
#endif
        // "find curves narrow band"
        cv::Mat idx1 = phi <= 1.2f;
        cv::Mat idx2 = phi >= -1.2f;
        cv::Mat idx = idx1 == idx2;
        // TODO: no narrow band, just everything
//        idx = cv::Mat::ones(phi.rows, phi.cols, CV_8U);

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
        float dt = .45f / ((float) maxdphidt + FLT_EPSILON); // 0.9*0.5 = 0.45
//        dt = .5f;

        cv::Mat dt_dphidt = dt * dphidt;
        cv::add(phi, dt_dphidt, phi, idx);

        sussmanReinit(phi, .5f);
#ifdef DEBUG
        clock2 = clock();
        std::cout << "Clock Diff: " << (clock2-clock1) << std::endl;
#endif
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

void RegBasedContours::sussmanReinit(cv::Mat& D, float dt)
{
    cv::Mat a; // D_x^-
    cv::Mat b; // D_x^+
    cv::Mat c; // D_y^-
    cv::Mat d; // D_y^+

    // calculate discretized derivates
    cv::hconcat(D.col(0), D.colRange(0, D.cols-1), a); // shift right
    cv::hconcat(D.colRange(1, D.cols), D.col(D.cols-1), b); // shift left
    cv::vconcat(D.row(0), D.rowRange(0, D.rows-1), c); // shift down
    cv::vconcat(D.rowRange(1, D.rows), D.row(D.rows-1), d); // shift up
    a = D - a;
    b = b - D;
    c = D - c;
    d = d - D;

    cv::Mat a_p = a.clone();
    cv::Mat b_p = b.clone();
    cv::Mat c_p = c.clone();
    cv::Mat d_p = d.clone();
    cv::Mat a_n = a.clone();
    cv::Mat b_n = b.clone();
    cv::Mat c_n = c.clone();
    cv::Mat d_n = d.clone();

    a_p.setTo(0, a < 0);
    a_n.setTo(0, a > 0);
    b_p.setTo(0, b < 0);
    b_n.setTo(0, b > 0);
    c_p.setTo(0, c < 0);
    c_n.setTo(0, c > 0);
    d_p.setTo(0, d < 0);
    d_n.setTo(0, d > 0);

    // calc S = D_ij / sqrt(D_ij.^2 + 1)
    cv::Mat S;
    cv::pow(D, 2, S);
    S = S + 1;
    cv::sqrt(S, S);
    cv::divide(D, S, S);

    // calc G
    cv::Mat G = cv::Mat::zeros(D.size(), D.type());
//    cv::Mat idx_pos = D > 0;
//    cv::Mat idx_neg = D < 0;

    cv::pow(a_p, 2, a_p);
    cv::pow(a_n, 2, a_n);
    cv::pow(b_p, 2, b_p);
    cv::pow(b_n, 2, b_n);
    cv::pow(c_p, 2, c_p);
    cv::pow(c_n, 2, c_n);
    cv::pow(d_p, 2, d_p);
    cv::pow(d_n, 2, d_n);

    cv::Mat max_ap_bn, max_cp_dn, max_an_bp, max_cn_dp;
    cv::max(a_p, b_n, max_ap_bn);
    cv::max(c_p, d_n, max_cp_dn);
    cv::max(a_n, b_p, max_an_bp);
    cv::max(c_n, d_p, max_cn_dp);

    cv::Mat sqrt_p, sqrt_n;
    cv::sqrt(max_ap_bn + max_cp_dn, sqrt_p);
    cv::sqrt(max_an_bp + max_cn_dp, sqrt_n);

    sqrt_p.copyTo(G, D > 0);
    sqrt_n.copyTo(G, D < 0);

    // calc new SDF
    cv::multiply(S, G, S);
    S = dt * S;
    D = D - S;
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
