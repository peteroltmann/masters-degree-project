#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

RegBasedContours::RegBasedContours(Method method, bool localized, int rad,
                                   float alpha) :
    method(method),
    localized(localized),
    rad(rad),
    alpha(alpha)
{}

RegBasedContours::~RegBasedContours() {}

void RegBasedContours::applySFM(cv::Mat& frame, cv::Mat init_mask,
                                int iterations)
{
#ifdef SAVE_AS_VIDEO
    cv::VideoWriter videoOut;
    videoOut.open("../output/output.avi", CV_FOURCC('X', 'V', 'I', 'D'), 60,
        frame.size(), false);
    if (!videoOut.isOpened())
    {
        std::cerr << "Could not write output video" << std::endl;
        return;
    }
#endif

    // =========================================================================
    // = INITIALIZATION                                                        =
    // =========================================================================

    if (frame.size() != init_mask.size())
    {
        std::cerr << "frame.size() != mask.size()" << std::endl;
        return;
    }

    set_frame(frame);
    init(init_mask);

    // =========================================================================
    // CONTOUR EVOLUTION                                                       =
    // =========================================================================

    for (int its = 0; its < iterations; its++)
    {
#ifdef TIME_MEASUREMENT
        int64 t1, t2;
        t1 = cv::getTickCount();
#endif

        iterate();

#ifdef TIME_MEASUREMENT
        t2 = cv::getTickCount();
        std::cout << "Time [s]: " << (t2-t1)/cv::getTickFrequency() << std::endl;
#endif
#ifdef SHOW_CONTOUR_EVOLUTION
        // show contours
        cv::Mat inOut = cv::Mat::zeros(image.rows, image.cols, image.type());
        inOut.setTo(255, phi < 0);
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Mat out;
        image.copyTo(out);
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 1);
        cv::imshow(WINDOW, out);
        cv::waitKey(1);

#ifdef SAVE_AS_VIDEO
        videoOut << out;
#endif
#endif
    }
#ifdef SAVE_AS_VIDEO
    videoOut.release();
#endif
}

void RegBasedContours::apply(cv::Mat frame, cv::Mat init_mask, int iterations)
{
#ifdef SHOW_CONTOUR_EVOLUTION
    cv::Mat image;
    frame.copyTo(image);
#endif
    frame.convertTo(frame, CV_32F);

    if (frame.size() != init_mask.size())
    {
        std::cerr << "frame.size() != mask.size()" << std::endl;
        return;
    }

    // create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask);

    // main loop
    for (int its = 0; its < iterations; its++)
    {
#ifdef TIME_MEASUREMENT
        int64 t1, t2;
        t1 = cv::getTickCount();
#endif
        // find the curve's narrow band, interior and exterior mean
        std::vector< std::vector<float> > narrow; // [y, x, value]
        float meanInt = 0.f, meanExt = 0.f;
        float sumInt = FLT_EPSILON, sumExt = FLT_EPSILON;
        for (int y = 0; y < phi.rows; y++)
        {
            const float* phiPtr = phi.ptr<float>(y);
            const float* framePtr = frame.ptr<float>(y);
            for (int x = 0; x < phi.cols; x++)
            {
                if (phiPtr[x] <= 1.2f && phiPtr[x] >= -1.2f)
                {
                    narrow.push_back(std::vector<float>(3));
                    narrow.back()[0] = y;
                    narrow.back()[1] = x;
                    narrow.back()[2] = 0.f;
                }
                if (!localized)
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
        }

        if (!localized)
        {
            meanInt /= sumInt;
            meanExt /= sumExt;
        }

        float maxF = 0.f;
        for (int i = 0; i < narrow.size(); i++)
        {
            int y = (int) narrow[i][0], x = (int) narrow[i][1];

            if (localized) // find localized mean
            {
                int xneg = x - rad < 0 ? 0 : x - rad;
                int xpos = x + rad > frame.cols-1 ? frame.cols-1 : x + rad;
                int yneg = y - rad < 0 ? 0 : y - rad;
                int ypos = y + rad > frame.rows-1 ? frame.rows-1 : y + rad;


                cv::Mat subImg = frame(cv::Rect(xneg, yneg, xpos-xneg+1,
                                                ypos-yneg+1));
                cv::Mat subPhi = phi(cv::Rect(xneg, yneg, xpos-xneg+1,
                                              ypos-yneg+1));

                meanInt = 0.f;
                meanExt = 0.f;
                sumInt = FLT_EPSILON;
                sumExt = FLT_EPSILON;
                for (int y = 0; y < subPhi.rows; y++)
                {
                    const float* subPhiPtr = subPhi.ptr<float>(y);
                    const float* subImgPtr = subImg.ptr<float>(y);
                    for (int x = 0; x < subPhi.cols; x++)
                    {
                        if (subPhiPtr[x] <= 0)
                        {
                            meanInt += subImgPtr[x];
                            sumInt++;
                        }
                        else
                        {
                            meanExt += subImgPtr[x];
                            sumExt++;
                        }
                    }
                }
                meanInt /= sumInt;
                meanExt /= sumExt;
            }

            // calculate speed WINDOW
            float Ix = frame.at<float>(y, x);
            float Fi = 0.f;

            // F = (I(x)-u).^2-(I(x)-v).^2
            if (method == CHAN_VESE)
            {
                float diffInt = Ix - meanInt;
                float diffExt = Ix - meanExt;
                Fi = diffInt*diffInt - diffExt*diffExt;
            }
            // F = -((u-v).*((I(idx)-u)./Ain+(I(idx)-v)./Aout));
            else if (method == YEZZI)
            {
                Fi = -((meanInt-meanExt) * ((Ix-meanInt) / sumInt
                                          + (Ix-meanExt) / sumExt));
            }

            narrow[i][2] = Fi;

            // get maxF for normalization
            float absFi = std::fabs(Fi);
            if (absFi > maxF)
                maxF = absFi;
        }

        // dphidt = F./max(abs(F)) + alpha*curvature;
        // extra loop to normalize speed WINDOW and calculate curvature
        for (int i = 0; i < narrow.size(); i++)
        {
            int y = (int) narrow[i][0], x = (int) narrow[i][1];

            // calculate curvature
            int xm1 = x == 0 ? 0 : x-1;
            int xp1 = x == phi.cols-1 ? phi.cols-1 : x+1;
            int ym1 = y == 0 ? 0 : y-1;
            int yp1 = y == phi.rows-1 ? phi.rows-1 : y+1;

            float phi_i = phi.at<float>(y, x);

            float phixx = (phi.at<float>(y, xp1)   - phi_i)
                        - (phi_i                   - phi.at<float>(y, xm1));
            float phiyy = (phi.at<float>(yp1, x)   - phi_i)
                        - (phi_i                   - phi.at<float>(ym1, x));
            float phixy = (phi.at<float>(yp1, xp1) - phi.at<float>(yp1, xm1))
                        - (phi.at<float>(ym1, xp1) - phi.at<float>(ym1, xm1));
            phixy *= 1.f/4.f;
            float phix = (phi.at<float>(y, xp1) - phi.at<float>(y, xm1));
            phix *= 1.f/.2f;
            float phiy = (phi.at<float>(yp1, x) - phi.at<float>(ym1, x));

            float curvature = (phixx*phiy*phiy
                               - 2.f*phiy*phix*phixy
                               + phiyy*phix*phix)
                              / std::pow((phix*phix + phiy*phiy + FLT_EPSILON),
                                         3.f/2.f);

            narrow[i][2] = narrow[i][2]/maxF + alpha*curvature;
        }

        maxF = FLT_MIN;
        for (int i = 0; i < narrow.size(); i++)
        {
            float Fi = narrow[i][2];
            if (Fi > maxF)
                maxF = Fi;
        }

        // maintain the CFL condition
        float dt = .45f / (maxF + FLT_EPSILON); // 0.9*0.5 = 0.45

        // phi = phi + dt * dphidt
        for (int i = 0; i < narrow.size(); i++)
        {
            int y = (int) narrow[i][0], x = (int) narrow[i][1];
            phi.at<float>(y, x) += dt * narrow[i][2];
        }

        sussman_reinit(phi, .5f);

#ifdef TIME_MEASUREMENT
        t2 = cv::getTickCount();
        std::cout << "Time [s]: " << (t2-t1)/cv::getTickFrequency() << std::endl;
#endif
#ifdef SHOW_CONTOUR_EVOLUTION
        // show contours
        cv::Mat inOut = cv::Mat::zeros(image.rows, image.cols, image.type());
        inOut.setTo(255, phi < 0);
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cv::Mat out;
        image.copyTo(out);
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 1);
        cv::imshow(WINDOW, out);
        cv::waitKey(1);
//        std::cout << "its: " << its << std::endl;
#endif
    }
}

void RegBasedContours::sussman_reinit(cv::Mat& D, float dt)
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
                Gptr[x] = std::sqrt(std::max(ap*ap, bn*bn) +
                                    std::max(cp*cp, dn*dn)) - 1;
            else if (Dx < 0)
                Gptr[x] = std::sqrt(std::max(an*an, bp*bp) +
                                    std::max(cn*cn, dp*dp)) - 1;
            else
                Gptr[x] = 0.f;

            // new SDF
            DnPtr[x] = Dx - dt * Sptr[x] * Gptr[x];
        }
    }
    D = Dn;
}

cv::Mat RegBasedContours::mask2phi(cv::Mat mask)
{
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

void RegBasedContours::set_frame(cv::Mat& frame)
{
    frame.copyTo(image);
    frame.convertTo(this->frame, CV_32F);
}

void RegBasedContours::set_params(Method method, bool localized, int rad,
                                 float alpha)
{
    this->method = method;
    this->localized = localized;
    this->rad = rad;
    this->alpha = alpha;
}

void RegBasedContours::init(cv::Mat& initMask)
{
    // reset level set lists
    lz.clear(); ln1.clear(); lp1.clear(); ln2.clear(); lp2.clear();
    phi = cv::Mat(frame.size(), frame.type(), cv::Scalar(-3));
    label = cv::Mat(phi.size(), cv::DataType<int>::type, cv::Scalar(-3));
    cv::Mat extPts = initMask == 0;
    label.setTo(3, extPts);
    phi.setTo(3, extPts);

    // find zero level set, consider bounds (-> index 3 to length-4)
    // and calculate global means
    sum_int = 0.f; sum_ext = 0.f;
    cnt_int = FLT_EPSILON; cnt_ext = FLT_EPSILON;
    mean_int = 0.f; mean_ext = 0.f;
    for (int y = 3; y < frame.rows-3; y++)
    {
        float* phiPtr = phi.ptr<float>(y);
        const float* framePtr = frame.ptr<float>(y);
        int* labelPtr = label.ptr<int>(y);
        const uchar* initMaskPtr = initMask.ptr<uchar>(y);
        for (int x = 3; x < frame.cols-4; x++)
        {
            uchar top = initMask.at<uchar>(y-1, x);
            uchar right = initMask.at<uchar>(y, x+1);
            uchar bottom = initMask.at<uchar>(y+1, x);
            uchar left = initMask.at<uchar>(y, x-1);
            if (initMaskPtr[x] == 1 &&
                (top == 0 || right == 0 || left == 0 || bottom == 0))
            {
                lz.push_back(cv::Point(x, y));
                labelPtr[x] = 0;
                phiPtr[x] = 0.f;
            }

            if (!localized)
            {
                if (phiPtr[x] <= 0)
                {
                    sum_int += framePtr[x];
                    cnt_int++;
                }
                else
                {
                    sum_ext += framePtr[x];
                    cnt_ext++;
                }
            }
        }
    }
    if (!localized)
    {
        mean_int = sum_int / cnt_int;
        mean_ext = sum_int / cnt_ext;
    }

    // find the +1 and -1 level set
    for (lz_it = lz.begin(); lz_it != lz.end(); lz_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = lz_it->y, x = lz_it->x;

        // for each neighbour
        for (int i = 0; i < 4; i++)
        {
            int y1 = 0; // y-neighbour
            int x1 = 0; // x-neighbour

            // set neighbour
            switch (i) {
                case 0: // top
                    y1 = y-1;
                    x1 = x;
                    break;
                case 1: // right
                    y1 = y;
                    x1 = x+1;
                    break;
                case 2: // bottom
                    y1 = y+1;
                    x1 = x;
                    break;
                case 3: // left
                    y1 = y;
                    x1 = x-1;
                    break;
                default:
                    break;
            }

            if (label.at<int>(y1, x1) == -3)
            {
                ln1.push_back(cv::Point(x1, y1));
                label.at<int>(y1, x1) = -1;
                phi.at<float>(y1, x1) = -1;
            }
            else if (label.at<int>(y1, x1) == 3)
            {
                lp1.push_back(cv::Point(x1, y1));
                label.at<int>(y1, x1) = 1;
                phi.at<float>(y1, x1) = 1;
            }

        }
    }

    // find the -2 level set
    for (ln1_it = ln1.begin(); ln1_it != ln1.end(); ln1_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = ln1_it->y, x = ln1_it->x;

        // for each neighbour
        for (int i = 0; i < 4; i++)
        {
            int y1 = 0; // y-neighbour
            int x1 = 0; // x-neighbour

            // set neighbour
            switch (i) {
                case 0: // top
                    y1 = y-1;
                    x1 = x;
                    break;
                case 1: // right
                    y1 = y;
                    x1 = x+1;
                    break;
                case 2: // bottom
                    y1 = y+1;
                    x1 = x;
                    break;
                case 3: // left
                    y1 = y;
                    x1 = x-1;
                    break;
                default:
                    break;
            }

            if (label.at<int>(y1, x1) == -3)
            {
                ln2.push_back(cv::Point(x1, y1));
                label.at<int>(y1, x1) = -2;
                phi.at<float>(y1, x1) = -2;
            }

        }
    }

    // find the +2 level set
    for (lp1_it = lp1.begin(); lp1_it != lp1.end(); lp1_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = lp1_it->y, x = lp1_it->x;

        // for each neighbour
        for (int i = 0; i < 4; i++)
        {
            int y1 = 0; // y-neighbour
            int x1 = 0; // x-neighbour

            // set neighbour
            switch (i) {
                case 0: // top
                    y1 = y-1;
                    x1 = x;
                    break;
                case 1: // right
                    y1 = y;
                    x1 = x+1;
                    break;
                case 2: // bottom
                    y1 = y+1;
                    x1 = x;
                    break;
                case 3: // left
                    y1 = y;
                    x1 = x-1;
                    break;
                default:
                    break;
            }

            if (label.at<int>(y1, x1) == 3)
            {
                lp2.push_back(cv::Point(x1, y1));
                label.at<int>(y1, x1) = 2;
                phi.at<float>(y1, x1) = 2;
            }

        }
    }
}

void RegBasedContours::iterate()
{
    // reset temporary lists
    sz.clear(); sn1.clear(); sp1.clear(); sn2.clear(); sp2.clear();
    std::list<cv::Point> lin, lout;
    std::list<cv::Point>::iterator lin_it, lout_it;

    calc_F(); // <-- Note: theoretically one iteration too much
             // (last normalization could be applied in the next loop)

    // update zero level set, actualize phi
    for (lz_it = lz.begin(); lz_it != lz.end(); lz_it++)
    {
        int y = lz_it->y, x = lz_it->x;

        float phixOld = phi.at<float>(y, x); // to check sign change

        // actualize phi
        float phix = phi.at<float>(y, x) += F.at<float>(y, x);

        if (phixOld <= 0 && phix > 0) // from inside to outside
        {
            lout.push_back(*lz_it);
        }
        else if (phixOld > 0 && phix <= 0) // from outside to inside
        {
            lin.push_back(*lz_it);
        }

        if (phix > .5f)
        {
            push_back(1, true, *lz_it, frame.size());
            lz_it = lz.erase(lz_it);
            lz_it--;
        }
        else if (phix < -.5f)
        {
            push_back(-1, true, *lz_it, frame.size());
            lz_it = lz.erase(lz_it);
            lz_it--;
        }
    }

    // update -1 level set
    for (ln1_it = ln1.begin(); ln1_it != ln1.end(); ln1_it++)
    {
        int y = ln1_it->y, x = ln1_it->x;

        int topL = label.at<int>(y-1, x);
        int rightL = label.at<int>(y, x+1);
        int bottomL = label.at<int>(y+1, x);
        int leftL = label.at<int>(y, x-1);

        if (topL != 0 && rightL != 0 && bottomL != 0 && leftL != 0)
        {
            push_back(-2, true, *ln1_it, frame.size());
            ln1_it = ln1.erase(ln1_it);
            ln1_it--;
        }
        else
        {
            float topPhi = phi.at<float>(y-1, x);
            float rightPhi = phi.at<float>(y, x+1);
            float bottomPhi = phi.at<float>(y+1, x);
            float leftPhi = phi.at<float>(y, x-1);

            // max phi of neighbours
            float max;
            max = topL >= 0 ? topPhi : -.5f; // -0.5 = min val of zero lvl
            max = rightL >= 0 && rightPhi > max ? rightPhi : max;
            max = bottomL >= 0 && bottomPhi > max ? bottomPhi : max;
            max = leftL >= 0 && leftPhi > max ? leftPhi : max;

            float phix = phi.at<float>(y, x) = max - 1.f;

            if (phix >= -.5f)
            {
                push_back(0, true, *ln1_it, frame.size());
               ln1_it = ln1.erase(ln1_it);
               ln1_it--;
            }
            else if (phix < -1.5f)
            {
                push_back(-2, true, *ln1_it, frame.size());
                ln1_it = ln1.erase(ln1_it);
                ln1_it--;
            }
        }
    }

    // update +1 level set
    for (lp1_it = lp1.begin(); lp1_it != lp1.end(); lp1_it++)
    {
        int y = lp1_it->y, x = lp1_it->x;

        int topL = label.at<int>(y-1, x);
        int rightL = label.at<int>(y, x+1);
        int bottomL = label.at<int>(y+1, x);
        int leftL = label.at<int>(y, x-1);

        if (topL != 0 && rightL != 0 && bottomL != 0 && leftL != 0)
        {
            push_back(2, true, *lp1_it, frame.size());
            lp1_it = lp1.erase(lp1_it);
            lp1_it--;
        }
        else
        {
            float topPhi = phi.at<float>(y-1, x);
            float rightPhi = phi.at<float>(y, x+1);
            float bottomPhi = phi.at<float>(y+1, x);
            float leftPhi = phi.at<float>(y, x-1);

            // min phi of neighbours
            float min;
            min = topL <= 0 ? topPhi : .5f; // 0.5 = max val of zero lvl
            min = rightL <= 0 && rightPhi < min ? rightPhi : min;
            min = bottomL <= 0 && bottomPhi < min ? bottomPhi : min;
            min = leftL <= 0 && leftPhi < min ? leftPhi : min;

            float phix = phi.at<float>(y, x) = min + 1.f;

            if (phix <= .5f)
            {
                push_back(0, true, *lp1_it, frame.size());
                lp1_it = lp1.erase(lp1_it);
                lp1_it--;
            }
            else if (phix > 1.5f)
            {
                push_back(2, true, *lp1_it, frame.size());
                lp1_it = lp1.erase(lp1_it);
                lp1_it--;
            }
        }
    }

    // update -2 level set
    for (ln2_it = ln2.begin(); ln2_it != ln2.end(); ln2_it++)
    {
        int y = ln2_it->y, x = ln2_it->x;

        int topL = label.at<int>(y-1, x);
        int rightL = label.at<int>(y, x+1);
        int bottomL = label.at<int>(y+1, x);
        int leftL = label.at<int>(y, x-1);

        if (topL != -1 && rightL != -1 && bottomL != -1 && leftL != -1)
        {
            ln2_it = ln2.erase(ln2_it);
            ln2_it--;
            label.at<int>(y, x) = -3;
            phi.at<float>(y, x) = -3.f;
        }
        else
        {
            float topPhi = phi.at<float>(y-1, x);
            float rightPhi = phi.at<float>(y, x+1);
            float bottomPhi = phi.at<float>(y+1, x);
            float leftPhi = phi.at<float>(y, x-1);

            // max phi of neighbours
            float max;
            max = topL >= -1 ? topPhi : -1.5f; // -1.5 = min val of -1 lvl
            max = rightL >= -1 && rightPhi > max ? rightPhi : max;
            max = bottomL >= -1 && bottomPhi > max ? bottomPhi : max;
            max = leftL >= -1 && leftPhi > max ? leftPhi : max;

            float phix = phi.at<float>(y, x) = max - 1.f;

            if (phix >= -1.5f)
            {
                push_back(-1, true, *ln2_it, frame.size());
                ln2_it = ln2.erase(ln2_it);
                ln2_it--;
            }
            else if (phix < -2.5f)
            {
                ln2_it = ln2.erase(ln2_it);
                ln2_it--;
                label.at<int>(y, x) = -3;
                phi.at<float>(y, x) = -3.f;
            }
        }
    }

    // update +2 level set
    for (lp2_it = lp2.begin(); lp2_it != lp2.end(); lp2_it++)
    {
        int y = lp2_it->y, x = lp2_it->x;

        int topL = label.at<int>(y-1, x);
        int rightL = label.at<int>(y, x+1);
        int bottomL = label.at<int>(y+1, x);
        int leftL = label.at<int>(y, x-1);

        if (topL != 1 && rightL != 1 && bottomL != 1 && leftL != 1)
        {
            lp2_it = lp2.erase(lp2_it);
            lp2_it--;
            label.at<int>(y, x) = 3;
            phi.at<float>(y, x) = 3.f;
        }
        else
        {
            float topPhi = phi.at<float>(y-1, x);
            float rightPhi = phi.at<float>(y, x+1);
            float bottomPhi = phi.at<float>(y+1, x);
            float leftPhi = phi.at<float>(y, x-1);

            // min phi of neighbours
            float min;
            min = topL <= 1 ? topPhi : 1.5f; // 1.5 = max val of 1 lvl
            min = rightL <= 1 && rightPhi < min ? rightPhi : min;
            min = bottomL <= 1 && bottomPhi < min ? bottomPhi : min;
            min = leftL <= 1 && leftPhi < min ? leftPhi : min;

            float phix = phi.at<float>(y, x) = min + 1.f;

            if (phix <= 1.5f)
            {
                push_back(1, true, *lp2_it, frame.size());
                lp2_it = lp2.erase(lp2_it);
                lp2_it--;
            }
            else if (phix > 2.5f)
            {
                lp2_it =lp2.erase(lp2_it);
                lp2_it--;
                label.at<int>(y, x) = 3;
                phi.at<float>(y, x) = 3.f;
            }
        }
    }

    // move points into zero level set
    for (sz_it = sz.begin(); sz_it != sz.end(); sz_it++)
    {
        int y = sz_it->y, x = sz_it->x;
        lz.push_back(*sz_it); // no bound check: already done for sz
        label.at<int>(y, x) = 0;
    }

    // move points into -1 level set and ensure -2 neighbours
    for (sn1_it = sn1.begin(); sn1_it != sn1.end(); sn1_it++)
    {
        int y = sn1_it->y, x = sn1_it->x;
        ln1.push_back(*sn1_it); // no bound check: already done for sn1
        label.at<int>(y, x) = -1;

        float phix = phi.at<float>(y, x);

        if (phi.at<float>(y-1, x) < -2.5f) // top
        {
            phi.at<float>(y-1, x) = phix - 1.f;
            push_back(-2, true, cv::Point(x, y-1), frame.size());
        }
        if (phi.at<float>(y, x+1) < -2.5f) // right
        {
            phi.at<float>(y, x+1) = phix - 1.f;
            push_back(-2, true, cv::Point(x+1, y), frame.size());
        }
        if (phi.at<float>(y+1, x) < -2.5f) // bottom
        {
            phi.at<float>(y+1, x) = phix - 1.f;
            push_back(-2, true, cv::Point(x, y+1), frame.size());
        }
        if (phi.at<float>(y, x-1) < -2.5f) // left
        {
            phi.at<float>(y, x-1) = phix - 1.f;
            push_back(-2, true, cv::Point(x-1, y), frame.size());
        }
    }

    // move points into +1 level set and ensure +2 neighbours
    for (sp1_it = sp1.begin(); sp1_it != sp1.end(); sp1_it++)
    {
        int y = sp1_it->y, x = sp1_it->x;
        lp1.push_back(*sp1_it); // no bound check: already done for sp1
        label.at<int>(y, x) = 1;

        float phix = phi.at<float>(y, x);

        if (phi.at<float>(y-1, x) > 2.5f) // top
        {
            phi.at<float>(y-1, x) = phix + 1.f;
            push_back(2, true, cv::Point(x, y-1), frame.size());
        }
        if (phi.at<float>(y, x+1) > 2.5f) // right
        {
            phi.at<float>(y, x+1) = phix + 1.f;
            push_back(2, true, cv::Point(x+1, y), frame.size());
        }
        if (phi.at<float>(y+1, x) > 2.5f) // bottom
        {
            phi.at<float>(y+1, x) = phix + 1.f;
            push_back(2, true, cv::Point(x, y+1), frame.size());
        }
        if (phi.at<float>(y, x-1) > 2.5f) // left
        {
            phi.at<float>(y, x-1) = phix + 1.f;
            push_back(2, true, cv::Point(x-1, y), frame.size());
        }
    }

    // move points into -2 level set
    for (sn2_it = sn2.begin(); sn2_it != sn2.end(); sn2_it++)
    {
        int y = sn2_it->y, x = sn2_it->x;
        ln2.push_back(*sn2_it); // no bound check: already done for sn2
        label.at<int>(y, x) = -2;
    }

    // move points into +2 level set
    for (sp2_it = sp2.begin(); sp2_it != sp2.end(); sp2_it++)
    {
        int y = sp2_it->y, x = sp2_it->x;
        lp2.push_back(*sp2_it); // no bound check: already done for sp2
        label.at<int>(y, x) = 2;
    }

    if (!localized)
    {
        // handle sign changes
        for (lin_it = lin.begin(); lin_it != lin.end(); lin_it++)
        {
            int y = lin_it->y, x = lin_it->x;
            float Ix = frame.at<float>(y, x);
            sum_int += Ix;
            sum_ext -= Ix;
            cnt_int++;
            cnt_ext--;
        }

        for (lout_it = lout.begin(); lout_it != lout.end(); lout_it++)
        {
            int y = lout_it->y, x = lout_it->x;
            float Ix = frame.at<float>(y, x);
            sum_int -= Ix;
            sum_ext += Ix;
            cnt_int--;
            cnt_ext++;
        }
        mean_int = sum_int / cnt_int;
        mean_ext = sum_ext / cnt_ext;
    }
}

void RegBasedContours::calc_F()
{
    F = cv::Mat(phi.size(), phi.type(), cv::Scalar(0));
    float maxF = 0.f;
    float maxF2 = 0.f;
    for (lz_it = lz.begin(); lz_it != lz.end(); lz_it++)
    {
        int y = lz_it->y, x = lz_it->x;

        if (localized) // find localized mean
        {
            int xneg = x - rad < 0 ? 0 : x - rad;
            int xpos = x + rad > frame.cols-1 ? frame.cols-1 : x + rad;
            int yneg = y - rad < 0 ? 0 : y - rad;
            int ypos = y + rad > frame.rows-1 ? frame.rows-1 : y + rad;

            cv::Mat subImg = frame(cv::Rect(xneg, yneg, xpos-xneg+1,
                                            ypos-yneg+1));
            cv::Mat subPhi = phi(cv::Rect(xneg, yneg, xpos-xneg+1,
                                          ypos-yneg+1));

            sum_int = 0.f; sum_ext = 0.f;
            cnt_int = FLT_EPSILON; cnt_ext = FLT_EPSILON;
            mean_int = 0.f; mean_ext = 0.f;
            for (int y = 0; y < subPhi.rows; y++)
            {
                const float* subPhiPtr = subPhi.ptr<float>(y);
                const float* subImgPtr = subImg.ptr<float>(y);
                for (int x = 0; x < subPhi.cols; x++)
                {
                    if (subPhiPtr[x] <= 0)
                    {
                        sum_int += subImgPtr[x];
                        cnt_int++;
                    }
                    else
                    {
                        sum_ext += subImgPtr[x];
                        cnt_ext++;
                    }
                }
            }
            mean_int = sum_int / cnt_int;
            mean_ext = sum_ext / cnt_ext;
        }

        // calculate speed WINDOW
        float Ix = frame.at<float>(y, x);
        float Fi = 0.f;

        // F = (I(x)-u).^2-(I(x)-v).^2
        if (method == CHAN_VESE)
        {
            float diffInt = Ix - mean_int;
            float diffExt = Ix - mean_ext;
            Fi = diffInt*diffInt - diffExt*diffExt;
        }
        // F = -((u-v).*((I(idx)-u)./Ain+(I(idx)-v)./Aout));
        else if (method == YEZZI)
        {
            Fi = -((mean_int - mean_ext) * ((Ix - mean_int) / cnt_int
                                          + (Ix - mean_ext) / cnt_ext));
        }

        F.at<float>(y, x) = Fi;

        // get maxF for normalization/scaling
        float absFi = std::fabs(Fi);
        if (absFi > maxF)
            maxF = absFi;
    }


    // dphidt = F./max(abs(F)) + alpha*curvature;
    // extra loop to normalize speed WINDOW and calc curvature
    for (lz_it = lz.begin(); lz_it != lz.end(); lz_it++)
    {
        int y = lz_it->y, x = lz_it->x;

        // calculate curvature
        int xm1 = x == 0 ? 0 : x-1;
        int xp1 = x == phi.cols-1 ? phi.cols-1 : x+1;
        int ym1 = y == 0 ? 0 : y-1;
        int yp1 = y == phi.rows-1 ? phi.rows-1 : y+1;

        float phi_i = phi.at<float>(y, x);

        float phixx = (phi.at<float>(y, xp1)   -  phi_i)
                    - ( phi_i                   - phi.at<float>(y, xm1));
        float phiyy = (phi.at<float>(yp1, x)   -  phi_i)
                    - ( phi_i                   - phi.at<float>(ym1, x));
        float phixy = (phi.at<float>(yp1, xp1) - phi.at<float>(yp1, xm1))
                    - (phi.at<float>(ym1, xp1) - phi.at<float>(ym1, xm1));
        phixy *= 1.f/4.f;
        float phix = (phi.at<float>(y, xp1) - phi.at<float>(y, xm1));
        phix *= 1.f/.2f;
        float phiy = (phi.at<float>(yp1, x) - phi.at<float>(ym1, x));

        float curvature = (phixx*phiy*phiy
                           - 2.f*phiy*phix*phixy
                           + phiyy*phix*phix)
                          / std::pow((phix*phix + phiy*phiy + FLT_EPSILON),
                                     3.f/2.f);

        // normalize/scale F, so curvature takes effect
        F.at<float>(y, x) = F.at<float>(y, x)/maxF + alpha*curvature;

        // find maxF (again) for normalization
        float absFi = std::fabs(F.at<float>(lz_it->y, lz_it->x));
        if (absFi > maxF2)
            maxF2 = absFi;
    }

    maxF = 0;
    for (lz_it = lz.begin(); lz_it != lz.end(); lz_it++)
    {
        int y = lz_it->y, x = lz_it->x;

        // normalize to |F| < 0.5
        F.at<float>(y, x) = (F.at<float>(y, x) / maxF2) *.45f;
    }
}

bool RegBasedContours::push_back(int listNo, bool tmp, cv::Point p, cv::Size size)
{
    int x = p.x;
    int y = p.y;
    bool success = false;
    switch (listNo)
    {
        case 0:
            if (x < 3 || x > size.width-4 || y < 3 || y > size.height-4)
                break;
            if (!tmp)
                lz.push_back(p);
            else
                sz.push_back(p);
            success = true;
            break;
        case -1:
            if (x < 2 || x > size.width-3 || y < 2 || y > size.height-3)
                break;
            if (!tmp)
                ln1.push_back(p);
            else
                sn1.push_back(p);
            success = true;
            break;
        case 1:
            if (x < 2 || x > size.width-3 || y < 2 || y > size.height-3)
                break;
            if (!tmp)
                lp1.push_back(p);
            else
                sp1.push_back(p);
            success = true;
            break;
        case -2:
            if (x < 1 || x > size.width-2 || y < 1 || y > size.height-2)
                break;
            if (!tmp)
                ln2.push_back(p);
            else
                sn2.push_back(p);
            success = true;
            break;
        case 2:
            if (x < 1 || x > size.width-2 || y < 1 || y > size.height-2)
                break;
            if (!tmp)
                lp2.push_back(p);
            else
                sp2.push_back(p);
            success = true;
            break;
        default:
            std::cerr << "List number out of range: " << listNo << std::endl;
            break;
    }
    return success;
}
