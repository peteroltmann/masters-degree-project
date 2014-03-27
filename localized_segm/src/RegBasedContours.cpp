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

        // find curves narrow band
        std::vector< std::vector<float> > narrow; // [y, x, value]
        for (int y = 0; y < phi.rows; y++)
        {
            const float* phiPtr = phi.ptr<float>(y);
            for (int x = 0; x < phi.cols; x++)
            {
                if (phiPtr[x] <= 1.2f && phiPtr[x] >= -1.2f)
                {
                    narrow.push_back(std::vector<float>(3));
                    narrow.back()[0] = y;
                    narrow.back()[1] = x;
                    narrow.back()[2] = 0.f;
                }
            }
        }

        // find interior and exterior mean
        float meanInt = 0.f, meanExt = 0.f;
        float sumInt = FLT_EPSILON, sumExt = FLT_EPSILON;
        for (int y = 0; y < phi.rows; y++)
        {
            const float* phiPtr = phi.ptr<float>(y);
            const float* framePtr = frame.ptr<float>(y);
            for (int x = 0; x < phi.cols; x++)
            {
                if (phiPtr[x] <= 0)
                {
                    meanInt += framePtr[x];
                    sumInt++;
                }
                else
                {
                    meanExt += framePtr[x];
                    sumExt++;
                }
            }
        }
        meanInt /= sumInt;
        meanExt /= sumExt;

        // F = (I(x)-u).^2-(I(x)-v).^2
        float maxF = 0.f;
        for (int i = 0; i < narrow.size(); i++)
        {
            int y = (int) narrow[i][0], x = (int) narrow[i][1];
            float Ix = frame.at<float>(y, x);
            float diffInt = Ix - meanInt;
            float diffExt = Ix - meanExt;
            float Fi = diffInt*diffInt - diffExt*diffExt;
            narrow[i][2] = Fi;

            if (std::fabs(Fi) > maxF)
                maxF = std::fabs(Fi);
        }

        // dphidt = F./max(abs(F)) + alpha*curvature;
        // % gradient descent to minimize energy
        // TODO: curvature
        for (int i = 0; i < narrow.size(); i++)
            narrow[i][2] /= maxF;

        maxF = FLT_MIN;
        for (int i = 0; i < narrow.size(); i++)
        {
            float Fi = narrow[i][2];
            if (Fi > maxF)
                maxF = Fi;
        }

        // % maintain the CFL condition
        // dt = .45/(max(dphidt)+eps);
        float dt = .45f / (maxF + FLT_EPSILON); // 0.9*0.5 = 0.45
//        dt = .5f;

        // phi = phi + dt * dphidt
        for (int i = 0; i < narrow.size(); i++)
        {
            int y = (int) narrow[i][0], x = (int) narrow[i][1];
            phi.at<float>(y, x) += dt * narrow[i][2];
        }

        sussmanReinit(phi, .5f);

#ifdef DEBUG
        clock2 = clock();
        std::cout << "Clock Diff: " << (clock2-clock1) << std::endl;
#endif
#ifdef SHOW_CONTOUR_EVOLUTION
        // show contours
        cv::Mat inOut = cv::Mat::zeros(image.rows, image.cols, image.type());
        inOut.setTo(255, phi < 0);
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Mat out;
        image.copyTo(out);
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Image", out);
        cv::waitKey(1);
//        std::cout << "its: " << its << std::endl;
#endif
    }
    seg = cv::Mat::zeros(image.rows, image.cols, image.type());
    seg.setTo(255, phi < 0);
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
    cv::Mat a(D.size(), D.type()); // D_x^-
    cv::Mat b(D.size(), D.type()); // D_x^+
    cv::Mat c(D.size(), D.type()); // D_y^-
    cv::Mat d(D.size(), D.type()); // D_y^+
    cv::Mat S(D.size(), D.type()); // S = D / (D.^2 + 1)
    cv::Mat G(D.size(), D.type());
    cv::Mat Dn(D.size(), D.type());

    // TOOD: what with the outer bound
    for (int y = 0; y < D.rows; y++)
    {
        const float* Dptr = D.ptr<float>(y);
        float* DnPtr = Dn.ptr<float>(y);
        float* aPtr = a.ptr<float>(y);
        float* bPtr = b.ptr<float>(y);
        float* cPtr = c.ptr<float>(y);
        float* dPtr = d.ptr<float>(y);
        float* Sptr = S.ptr<float>(y);
        float* Gptr = G.ptr<float>(y);
        for (int x = 0; x < D.cols; x++)
        {
            float Dx = Dptr[x];

            int xm1 = x == 0 ? 0 : x-1;
            int xp1 = x == D.cols-1 ? D.cols-1 : x+1;
            int ym1 = y == 0 ? 0 : y-1;
            int yp1 = y == D.rows-1 ? D.rows-1 : y+1;

            // calculate discretized derivates
            aPtr[x] = (Dx - D.at<float>(y, xm1));
            bPtr[x] = (D.at<float>(y, xp1) - Dx);
            cPtr[x] = (Dx - D.at<float>(ym1, x));
            dPtr[x] = (D.at<float>(yp1, x) - Dx);

            Sptr[x] = Dx / std::sqrt(Dx*Dx + 1);

            // positive/negative values
            float ap = aPtr[x] < 0 ? 0 : aPtr[x];
            float an = aPtr[x] > 0 ? 0 : aPtr[x];
            float bp = bPtr[x] < 0 ? 0 : bPtr[x];
            float bn = bPtr[x] > 0 ? 0 : bPtr[x];
            float cp = cPtr[x] < 0 ? 0 : cPtr[x];
            float cn = cPtr[x] > 0 ? 0 : cPtr[x];
            float dp = dPtr[x] < 0 ? 0 : dPtr[x];
            float dn = dPtr[x] > 0 ? 0 : dPtr[x];

            if (Dx > 0)
                Gptr[x] = std::sqrt(std::max(ap*ap, bn*bn) + std::max(cp*cp, dn*dn)) - 1;
            else if (Dx < 0)
                Gptr[x] = std::sqrt(std::max(an*an, bp*bp) + std::max(cn*cn, dp*dp)) - 1;
            else
                Gptr[x] = 0.f;

            // new SDF
            DnPtr[x] = Dx - dt * Sptr[x] * Gptr[x];
        }
    }
    Dn.copyTo(D);
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
