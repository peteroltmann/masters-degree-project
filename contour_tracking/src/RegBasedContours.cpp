#include "RegBasedContours.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/format.hpp>

RegBasedContours::RegBasedContours() :
    _localized(false),
    _method(CHAN_VESE),
    _alpha(.2f),
    _rad(18.f)
{}

RegBasedContours::~RegBasedContours() {}

void RegBasedContours::applySFM(cv::Mat& frame, cv::Mat initMask,
                                int iterations, int method, bool localized,
                                int rad, float alpha)
{
#ifdef SAVE_AS_VIDEO
    cv::VideoWriter videoOut;
    videoOut.open("C:/Users/Peter/Desktop/output.avi", -1, 60, frame.size(), false);
    if (!videoOut.isOpened())
    {
        std::cerr << "Could not write output video" << std::endl;
        return;
    }
#endif

    // =========================================================================
    // = INITIALIZATION                                                        =
    // =========================================================================

    _method = method;
    _localized = localized;
    _rad = rad;
    _alpha = alpha;

    setFrame(frame);
    init(initMask);

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
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 2);
        cv::imshow(WINDOW_NAME, out);
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

void RegBasedContours::apply(cv::Mat frame, cv::Mat initMask, cv::Mat& phi,
                             int iterations, int method, bool localized,
                             int rad, float alpha)
{
#ifdef SHOW_CONTOUR_EVOLUTION
    cv::Mat image;
    frame.copyTo(image);
#endif
    frame.convertTo(frame, CV_32F);

    // TODO assert frame.size() = initMask.size()

    // create a signed distance map (SDF) from mask
    phi = mask2phi(initMask);

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

            // calculate speed function
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
        // extra loop to normalize speed function and calculate curvature
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

        sussmanReinit(phi, .5f);

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
        cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 2);
        cv::imshow(WINDOW_NAME, out);
        cv::waitKey(1);
//        std::cout << "its: " << its << std::endl;
#endif
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

void RegBasedContours::setFrame(cv::Mat& frame)
{
    frame.copyTo(_image);
    frame.convertTo(_frame, CV_32F);
}

void RegBasedContours::init(cv::Mat& initMask)
{
    // reset level set lists
    _lz.clear(); _ln1.clear(); _lp1.clear(); _ln2.clear(); _lp2.clear();
    _phi = cv::Mat(_frame.size(), _frame.type(), cv::Scalar(-3));
    _label = cv::Mat(_phi.size(), cv::DataType<int>::type, cv::Scalar(-3));
    cv::Mat extPts = initMask == 0;
    _label.setTo(3, extPts);
    _phi.setTo(3, extPts);

    // find zero level set, consider bounds (-> index 3 to length-4)
    // and calculate global means
    _sumInt = 0.f; _sumExt = 0.f;
    _cntInt = FLT_EPSILON; _cntExt = FLT_EPSILON;
    _meanInt = 0.f; _meanExt = 0.f;
    for (int y = 3; y < _frame.rows-3; y++)
    {
        float* phiPtr = _phi.ptr<float>(y);
        const float* framePtr = _frame.ptr<float>(y);
        int* labelPtr = _label.ptr<int>(y);
        const uchar* initMaskPtr = initMask.ptr<uchar>(y);
        for (int x = 3; x < _frame.cols-4; x++)
        {
            uchar top = initMask.at<uchar>(y-1, x);
            uchar right = initMask.at<uchar>(y, x+1);
            uchar bottom = initMask.at<uchar>(y+1, x);
            uchar left = initMask.at<uchar>(y, x-1);
            if (initMaskPtr[x] == 1 &&
                (top == 0 || right == 0 || left == 0 || bottom == 0))
            {
                _lz.push_back(cv::Point(x, y));
                labelPtr[x] = 0;
                phiPtr[x] = 0.f;
            }

            if (!_localized)
            {
                if (phiPtr[x] <= 0)
                {
                    _sumInt += framePtr[x];
                    _cntInt++;
                }
                else
                {
                    _sumExt += framePtr[x];
                    _cntExt++;
                }
            }
        }
    }
    if (!_localized)
    {
        _meanInt = _sumInt / _cntInt;
        _meanExt = _sumInt / _cntExt;
    }

    // find the +1 and -1 level set
    for (_lz_it = _lz.begin(); _lz_it != _lz.end(); _lz_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = _lz_it->y, x = _lz_it->x;

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

            if (_label.at<int>(y1, x1) == -3)
            {
                _ln1.push_back(cv::Point(x1, y1));
                _label.at<int>(y1, x1) = -1;
                _phi.at<float>(y1, x1) = -1;
            }
            else if (_label.at<int>(y1, x1) == 3)
            {
                _lp1.push_back(cv::Point(x1, y1));
                _label.at<int>(y1, x1) = 1;
                _phi.at<float>(y1, x1) = 1;
            }

        }
    }

    // find the -2 level set
    for (_ln1_it = _ln1.begin(); _ln1_it != _ln1.end(); _ln1_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = _ln1_it->y, x = _ln1_it->x;

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

            if (_label.at<int>(y1, x1) == -3)
            {
                _ln2.push_back(cv::Point(x1, y1));
                _label.at<int>(y1, x1) = -2;
                _phi.at<float>(y1, x1) = -2;
            }

        }
    }

    // find the +2 level set
    for (_lp1_it = _lp1.begin(); _lp1_it != _lp1.end(); _lp1_it++)
    {
        // no bound check, because bound pixels werde not considered
        int y = _lp1_it->y, x = _lp1_it->x;

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

            if (_label.at<int>(y1, x1) == 3)
            {
                _lp2.push_back(cv::Point(x1, y1));
                _label.at<int>(y1, x1) = 2;
                _phi.at<float>(y1, x1) = 2;
            }

        }
    }
}

void RegBasedContours::iterate()
{
    // reset temporary lists
    _sz.clear(); _sn1.clear(); _sp1.clear(); _sn2.clear(); _sp2.clear();
    std::list<cv::Point> lin, lout;
    std::list<cv::Point>::iterator lin_it, lout_it;

    calcF(); // <-- Note: theoretically one iteration too much
             // (last normalization could be applied in the next loop)

    // update zero level set, actualize phi
    for (_lz_it = _lz.begin(); _lz_it != _lz.end(); _lz_it++)
    {
        int y = _lz_it->y, x = _lz_it->x;

        float phixOld = _phi.at<float>(y, x); // to check sign change

        // actualize phi
        float phix = _phi.at<float>(y, x) += _F.at<float>(y, x);

        if (phixOld <= 0 && phix > 0) // from inside to outside
        {
            lout.push_back(*_lz_it);
        }
        else if (phixOld > 0 && phix <= 0) // from outside to inside
        {
            lin.push_back(*_lz_it);
        }

        if (phix > .5f)
        {
            pushBack(1, true, *_lz_it, _frame.size());
            _lz.erase(_lz_it);
        }
        else if (phix < -.5f)
        {
            pushBack(-1, true, *_lz_it, _frame.size());
            _lz.erase(_lz_it);
        }
    }

    // update -1 level set
    for (_ln1_it = _ln1.begin(); _ln1_it != _ln1.end(); _ln1_it++)
    {
        int y = _ln1_it->y, x = _ln1_it->x;

        int topL = _label.at<int>(y-1, x);
        int rightL = _label.at<int>(y, x+1);
        int bottomL = _label.at<int>(y+1, x);
        int leftL = _label.at<int>(y, x-1);

        if (topL != 0 && rightL != 0 && bottomL != 0 && leftL != 0)
        {
            pushBack(-2, true, *_ln1_it, _frame.size());
            _ln1.erase(_ln1_it);
        }
        else
        {
            float topPhi = _phi.at<float>(y-1, x);
            float rightPhi = _phi.at<float>(y, x+1);
            float bottomPhi = _phi.at<float>(y+1, x);
            float leftPhi = _phi.at<float>(y, x-1);

            // max phi of neighbours
            float max;
            max = topL >= 0 ? topPhi : -.5f; // -0.5 = min val of zero lvl
            max = rightL >= 0 && rightPhi > max ? rightPhi : max;
            max = bottomL >= 0 && bottomPhi > max ? bottomPhi : max;
            max = leftL >= 0 && leftPhi > max ? leftPhi : max;

            float phix = _phi.at<float>(y, x) = max - 1.f;

            if (phix >= -.5f)
            {
                pushBack(0, true, *_ln1_it, _frame.size());
               _ln1.erase(_ln1_it);
            }
            else if (phix < -1.5f)
            {
                pushBack(-2, true, *_ln1_it, _frame.size());
                _ln1.erase(_ln1_it);
            }
        }
    }

    // update +1 level set
    for (_lp1_it = _lp1.begin(); _lp1_it != _lp1.end(); _lp1_it++)
    {
        int y = _lp1_it->y, x = _lp1_it->x;

        int topL = _label.at<int>(y-1, x);
        int rightL = _label.at<int>(y, x+1);
        int bottomL = _label.at<int>(y+1, x);
        int leftL = _label.at<int>(y, x-1);

        if (topL != 0 && rightL != 0 && bottomL != 0 && leftL != 0)
        {
            pushBack(2, true, *_lp1_it, _frame.size());
            _lp1.erase(_lp1_it);
        }
        else
        {
            float topPhi = _phi.at<float>(y-1, x);
            float rightPhi = _phi.at<float>(y, x+1);
            float bottomPhi = _phi.at<float>(y+1, x);
            float leftPhi = _phi.at<float>(y, x-1);

            // min phi of neighbours
            float min;
            min = topL <= 0 ? topPhi : .5f; // 0.5 = max val of zero lvl
            min = rightL <= 0 && rightPhi < min ? rightPhi : min;
            min = bottomL <= 0 && bottomPhi < min ? bottomPhi : min;
            min = leftL <= 0 && leftPhi < min ? leftPhi : min;

            float phix = _phi.at<float>(y, x) = min + 1.f;

            if (phix <= .5f)
            {
                pushBack(0, true, *_lp1_it, _frame.size());
                _lp1.erase(_lp1_it);
            }
            else if (phix > 1.5f)
            {
                pushBack(2, true, *_lp1_it, _frame.size());
                _lp1.erase(_lp1_it);
            }
        }
    }

    // update -2 level set
    for (_ln2_it = _ln2.begin(); _ln2_it != _ln2.end(); _ln2_it++)
    {
        int y = _ln2_it->y, x = _ln2_it->x;

        int topL = _label.at<int>(y-1, x);
        int rightL = _label.at<int>(y, x+1);
        int bottomL = _label.at<int>(y+1, x);
        int leftL = _label.at<int>(y, x-1);

        if (topL != -1 && rightL != -1 && bottomL != -1 && leftL != -1)
        {
            _ln2.erase(_ln2_it);
            _label.at<int>(y, x) = -3;
            _phi.at<float>(y, x) = -3.f;
        }
        else
        {
            float topPhi = _phi.at<float>(y-1, x);
            float rightPhi = _phi.at<float>(y, x+1);
            float bottomPhi = _phi.at<float>(y+1, x);
            float leftPhi = _phi.at<float>(y, x-1);

            // max phi of neighbours
            float max;
            max = topL >= -1 ? topPhi : -1.5f; // -1.5 = min val of -1 lvl
            max = rightL >= -1 && rightPhi > max ? rightPhi : max;
            max = bottomL >= -1 && bottomPhi > max ? bottomPhi : max;
            max = leftL >= -1 && leftPhi > max ? leftPhi : max;

            float phix = _phi.at<float>(y, x) = max - 1.f;

            if (phix >= -1.5f)
            {
                pushBack(-1, true, *_ln2_it, _frame.size());
                _ln2.erase(_ln2_it);
            }
            else if (phix < -2.5f)
            {
                _ln2.erase(_ln2_it);
                _label.at<int>(y, x) = -3;
                _phi.at<float>(y, x) = -3.f;
            }
        }
    }

    // update +2 level set
    for (_lp2_it = _lp2.begin(); _lp2_it != _lp2.end(); _lp2_it++)
    {
        int y = _lp2_it->y, x = _lp2_it->x;

        int topL = _label.at<int>(y-1, x);
        int rightL = _label.at<int>(y, x+1);
        int bottomL = _label.at<int>(y+1, x);
        int leftL = _label.at<int>(y, x-1);

        if (topL != 1 && rightL != 1 && bottomL != 1 && leftL != 1)
        {
            _lp2.erase(_lp2_it);
            _label.at<int>(y, x) = 3;
            _phi.at<float>(y, x) = 3.f;
        }
        else
        {
            float topPhi = _phi.at<float>(y-1, x);
            float rightPhi = _phi.at<float>(y, x+1);
            float bottomPhi = _phi.at<float>(y+1, x);
            float leftPhi = _phi.at<float>(y, x-1);

            // min phi of neighbours
            float min;
            min = topL <= 1 ? topPhi : 1.5f; // 1.5 = max val of 1 lvl
            min = rightL <= 1 && rightPhi < min ? rightPhi : min;
            min = bottomL <= 1 && bottomPhi < min ? bottomPhi : min;
            min = leftL <= 1 && leftPhi < min ? leftPhi : min;

            float phix = _phi.at<float>(y, x) = min + 1.f;

            if (phix <= 1.5f)
            {
                pushBack(1, true, *_lp2_it, _frame.size());
                _lp2.erase(_lp2_it);
            }
            else if (phix > 2.5f)
            {
                _lp2.erase(_lp2_it);
                _label.at<int>(y, x) = 3;
                _phi.at<float>(y, x) = 3.f;
            }
        }
    }

    // move points into zero level set
    for (_sz_it = _sz.begin(); _sz_it != _sz.end(); _sz_it++)
    {
        int y = _sz_it->y, x = _sz_it->x;
        _lz.push_back(*_sz_it); // no bound check: already done for sz
        _label.at<int>(y, x) = 0;
    }

    // move points into -1 level set and ensure -2 neighbours
    for (_sn1_it = _sn1.begin(); _sn1_it != _sn1.end(); _sn1_it++)
    {
        int y = _sn1_it->y, x = _sn1_it->x;
        _ln1.push_back(*_sn1_it); // no bound check: already done for sn1
        _label.at<int>(y, x) = -1;

        float phix = _phi.at<float>(y, x);

        if (_phi.at<float>(y-1, x) < -2.5f) // top
        {
            _phi.at<float>(y-1, x) = phix - 1.f;
            pushBack(-2, true, cv::Point(x, y-1), _frame.size());
        }
        if (_phi.at<float>(y, x+1) < -2.5f) // right
        {
            _phi.at<float>(y, x+1) = phix - 1.f;
            pushBack(-2, true, cv::Point(x+1, y), _frame.size());
        }
        if (_phi.at<float>(y+1, x) < -2.5f) // bottom
        {
            _phi.at<float>(y+1, x) = phix - 1.f;
            pushBack(-2, true, cv::Point(x, y+1), _frame.size());
        }
        if (_phi.at<float>(y, x-1) < -2.5f) // left
        {
            _phi.at<float>(y, x-1) = phix - 1.f;
            pushBack(-2, true, cv::Point(x-1, y), _frame.size());
        }
    }

    // move points into +1 level set and ensure +2 neighbours
    for (_sp1_it = _sp1.begin(); _sp1_it != _sp1.end(); _sp1_it++)
    {
        int y = _sp1_it->y, x = _sp1_it->x;
        _lp1.push_back(*_sp1_it); // no bound check: already done for sp1
        _label.at<int>(y, x) = 1;

        float phix = _phi.at<float>(y, x);

        if (_phi.at<float>(y-1, x) > 2.5f) // top
        {
            _phi.at<float>(y-1, x) = phix + 1.f;
            pushBack(2, true, cv::Point(x, y-1), _frame.size());
        }
        if (_phi.at<float>(y, x+1) > 2.5f) // right
        {
            _phi.at<float>(y, x+1) = phix + 1.f;
            pushBack(2, true, cv::Point(x+1, y), _frame.size());
        }
        if (_phi.at<float>(y+1, x) > 2.5f) // bottom
        {
            _phi.at<float>(y+1, x) = phix + 1.f;
            pushBack(2, true, cv::Point(x, y+1), _frame.size());
        }
        if (_phi.at<float>(y, x-1) > 2.5f) // left
        {
            _phi.at<float>(y, x-1) = phix + 1.f;
            pushBack(2, true, cv::Point(x-1, y), _frame.size());
        }
    }

    // move points into -2 level set
    for (_sn2_it = _sn2.begin(); _sn2_it != _sn2.end(); _sn2_it++)
    {
        int y = _sn2_it->y, x = _sn2_it->x;
        _ln2.push_back(*_sn2_it); // no bound check: already done for sn2
        _label.at<int>(y, x) = -2;
    }

    // move points into +2 level set
    for (_sp2_it = _sp2.begin(); _sp2_it != _sp2.end(); _sp2_it++)
    {
        int y = _sp2_it->y, x = _sp2_it->x;
        _lp2.push_back(*_sp2_it); // no bound check: already done for sp2
        _label.at<int>(y, x) = 2;
    }

    if (!_localized)
    {
        // handle sign changes
        for (lin_it = lin.begin(); lin_it != lin.end(); lin_it++)
        {
            int y = lin_it->y, x = lin_it->x;
            float Ix = _frame.at<float>(y, x);
            _sumInt += Ix;
            _sumExt -= Ix;
            _cntInt++;
            _cntExt--;
        }

        for (lout_it = lout.begin(); lout_it != lout.end(); lout_it++)
        {
            int y = lout_it->y, x = lout_it->x;
            float Ix = _frame.at<float>(y, x);
            _sumInt -= Ix;
            _sumExt += Ix;
            _cntInt--;
            _cntExt++;
        }
        _meanInt = _sumInt / _cntInt;
        _meanExt = _sumExt / _cntExt;
    }
}

void RegBasedContours::calcF()
{
    _F = cv::Mat(_phi.size(), _phi.type(), cv::Scalar(0));
    float maxF = 0.f;
    float maxF2 = 0.f;
    for (_lz_it = _lz.begin(); _lz_it != _lz.end(); _lz_it++)
    {
        int y = _lz_it->y, x = _lz_it->x;

        if (_localized) // find localized mean
        {
            int xneg = x - _rad < 0 ? 0 : x - _rad;
            int xpos = x + _rad > _frame.cols-1 ? _frame.cols-1 : x + _rad;
            int yneg = y - _rad < 0 ? 0 : y - _rad;
            int ypos = y + _rad > _frame.rows-1 ? _frame.rows-1 : y + _rad;

            cv::Mat subImg = _frame(cv::Rect(xneg, yneg, xpos-xneg+1,
                                            ypos-yneg+1));
            cv::Mat subPhi = _phi(cv::Rect(xneg, yneg, xpos-xneg+1,
                                          ypos-yneg+1));

            _sumInt = 0.f; _sumExt = 0.f;
            _cntInt = FLT_EPSILON; _cntExt = FLT_EPSILON;
            _meanInt = 0.f; _meanExt = 0.f;
            for (int y = 0; y < subPhi.rows; y++)
            {
                const float* subPhiPtr = subPhi.ptr<float>(y);
                const float* subImgPtr = subImg.ptr<float>(y);
                for (int x = 0; x < subPhi.cols; x++)
                {
                    if (subPhiPtr[x] <= 0)
                    {
                        _sumInt += subImgPtr[x];
                        _cntInt++;
                    }
                    else
                    {
                        _sumExt += subImgPtr[x];
                        _cntExt++;
                    }
                }
            }
            _meanInt = _sumInt / _cntInt;
            _meanExt = _sumExt / _cntExt;
        }

        // calculate speed function
        float Ix = _frame.at<float>(y, x);
        float Fi = 0.f;

        // F = (I(x)-u).^2-(I(x)-v).^2
        if (_method == CHAN_VESE)
        {
            float diffInt = Ix - _meanInt;
            float diffExt = Ix - _meanExt;
            Fi = diffInt*diffInt - diffExt*diffExt;
        }
        // F = -((u-v).*((I(idx)-u)./Ain+(I(idx)-v)./Aout));
        else if (_method == YEZZI)
        {
            Fi = -((_meanInt - _meanExt) * ((Ix - _meanInt) / _cntInt
                                          + (Ix - _meanExt) / _cntExt));
        }

        _F.at<float>(y, x) = Fi;

        // get maxF for normalization/scaling
        float absFi = std::fabs(Fi);
        if (absFi > maxF)
            maxF = absFi;
    }


    // dphidt = F./max(abs(F)) + alpha*curvature;
    // extra loop to normalize speed function and calc curvature
    for (_lz_it = _lz.begin(); _lz_it != _lz.end(); _lz_it++)
    {
        int y = _lz_it->y, x = _lz_it->x;

        // calculate curvature
        int xm1 = x == 0 ? 0 : x-1;
        int xp1 = x == _phi.cols-1 ? _phi.cols-1 : x+1;
        int ym1 = y == 0 ? 0 : y-1;
        int yp1 = y == _phi.rows-1 ? _phi.rows-1 : y+1;

        float phi_i = _phi.at<float>(y, x);

        float phixx = (_phi.at<float>(y, xp1)   -  phi_i)
                    - ( phi_i                   - _phi.at<float>(y, xm1));
        float phiyy = (_phi.at<float>(yp1, x)   -  phi_i)
                    - ( phi_i                   - _phi.at<float>(ym1, x));
        float phixy = (_phi.at<float>(yp1, xp1) - _phi.at<float>(yp1, xm1))
                    - (_phi.at<float>(ym1, xp1) - _phi.at<float>(ym1, xm1));
        phixy *= 1.f/4.f;
        float phix = (_phi.at<float>(y, xp1) - _phi.at<float>(y, xm1));
        phix *= 1.f/.2f;
        float phiy = (_phi.at<float>(yp1, x) - _phi.at<float>(ym1, x));

        float curvature = (phixx*phiy*phiy
                           - 2.f*phiy*phix*phixy
                           + phiyy*phix*phix)
                          / std::pow((phix*phix + phiy*phiy + FLT_EPSILON),
                                     3.f/2.f);

        // normalize/scale F, so curvature takes effect
        _F.at<float>(y, x) = _F.at<float>(y, x)/maxF + _alpha*curvature;

        // find maxF (again) for normalization
        float absFi = std::fabs(_F.at<float>(_lz_it->y, _lz_it->x));
        if (absFi > maxF2)
            maxF2 = absFi;
    }

    maxF = 0;
    for (_lz_it = _lz.begin(); _lz_it != _lz.end(); _lz_it++)
    {
        int y = _lz_it->y, x = _lz_it->x;

        // normalize to |F| < 0.5
        _F.at<float>(y, x) = (_F.at<float>(y, x) / maxF2) *.45f;
    }
}

bool RegBasedContours::pushBack(int listNo, bool tmp, cv::Point p, cv::Size size)
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
                _lz.push_back(p);
            else
                _sz.push_back(p);
            success = true;
            break;
        case -1:
            if (x < 2 || x > size.width-3 || y < 2 || y > size.height-3)
                break;
            if (!tmp)
                _ln1.push_back(p);
            else
                _sn1.push_back(p);
            success = true;
            break;
        case 1:
            if (x < 2 || x > size.width-3 || y < 2 || y > size.height-3)
                break;
            if (!tmp)
                _lp1.push_back(p);
            else
                _sp1.push_back(p);
            success = true;
            break;
        case -2:
            if (x < 1 || x > size.width-2 || y < 1 || y > size.height-2)
                break;
            if (!tmp)
                _ln2.push_back(p);
            else
                _sn2.push_back(p);
            success = true;
            break;
        case 2:
            if (x < 1 || x > size.width-2 || y < 1 || y > size.height-2)
                break;
            if (!tmp)
                _lp2.push_back(p);
            else
                _sp2.push_back(p);
            success = true;
            break;
        default:
            std::cerr << "List number out of range: " << listNo << std::endl;
            break;
    }
}
