#include "FourierDescriptor.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

FourierDescriptor::FourierDescriptor(const cv::Mat_<uchar>& mask,
                                     int num_samples)
{
    init(mask, num_samples);
}

FourierDescriptor::~FourierDescriptor() {}

void FourierDescriptor::init(const cv::Mat_<uchar>& mask, int num_samples)
{
    this->num_samples = num_samples;
//    U = cv::Mat_<cv::Vec2f>(num_samples, 1);
    U.release();

    // calculate mass center
    cv::Moments m = cv::moments(mask, true);
    new (&center) cv::Point(m.m10/m.m00, m.m01/m.m00);

    // estimate outer contour(s)
    outline_mask = mask == 1;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(outline_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    outline_mask.setTo(0);
    cv::drawContours(outline_mask, contours, -1, 1);

//    sort(); // sort contour points --> cp
    cp = contours[0];
    std::reverse(cp.begin(), cp.end());

//    cv::Mat asd;
//    outline_mask.copyTo(asd);
//    for (int i = 0, j = 0; i < cp.size(); i+=cp.size()/num_samples, j++)
//    {
//        cv::circle(asd, cp[i], 0, 255);
//        cv::imshow("YO2", asd);
//        cv::waitKey();
//    }

    // create complex vector
    float arclen = cv::arcLength(cp, true);
    float delta = arclen / num_samples;
    std::cout << "delta: " << delta << std::endl;

    int i = 0;
    float sub_arclen = 0.f;
    float sub_arclen_next = 0.f;

    U.push_back(cv::Vec2f(cp[0].x, cp[0].y));

    while(i < cp.size()-1)
    {
        while (sub_arclen_next <= delta && i < cp.size()-1)
        {
            sub_arclen_next += cv::norm(cp[i+1] - cp[i]);
//            std::cout << sub_arclen_next << std::endl;
            if (sub_arclen_next <= delta)
            {
                sub_arclen = sub_arclen_next;
                i++;
            }
//            std::cout << sub_arclen << std::endl;
        }

        if (i >= cp.size()-1)
            break;

        float d = delta - sub_arclen;
        cv:Values inside the function:  [1, 2, 3, 4]:Vec2f p = cv::Vec2f(cp[i].x, cp[i].y);
        cv::Vec2f v = cv::Vec2f(cp[i+1].x, cp[i+1].y);
        v = v - p;
        v = v / cv::norm(v);
        p = p + d*v;

        U.push_back(p);

        sub_arclen_next = cv::norm(cv::Vec2f(cp[i+1].x, cp[i+1].y) - p);
        std::cout << d << " - " << sub_arclen_next + d << std::endl;
//        std::cout << sub_arclen + d << std::endl;
//        sub_arclen_next = delta - d;
//        sub_arclen_next = 0.f;
        i++;
    }

//    cv::Mat asd;
//    outline_mask.copyTo(asd);
//    for (int i = 0; i < U.total(); i++)
//    {
//        cv::circle(asd, cv::Point(U(i)[0], U(i)[1]), 0, 255);
//        cv::imshow("YO2", asd);
//        cv::waitKey();
//    }

//    for (int i = 0; i < U.total()-1; i++)
//    {
//        std::cout << norm(U(i+1), U(i)) << std::endl;
//    }
    std::cout << U.size() << std::endl;

    num_points = cp.size();
//    z = std::vector<std::complex<float>>(num_points);
//    for (int i = 0; i < cp.size(); i++)
//    {
//        z[i] = std::complex<float>(cp[i].x, cp[i].y);
//    }

//    float idx = 0;
//    z = std::vector<std::complex<float>>(num_samples);
//    for (int i = 0; i < num_samples; idx += delta, i++)
//    {
//        int j = std::round(idx);
//        z[i] = std::complex<float>(cp[j].x, cp[j].y);
//    }

//    dft(num_samples, num_samples/2);
////    for (int k = 0; k < c.size(); k++)
////    {
////        std::cout << c[k] << std::endl;
////    }

//    idft(num_samples, num_samples/2);
////    for (int i = 0; i < iz.size(); i++)
////    {
////        std::cout << iz[i] << std::endl;
////    }

//    cv::Mat reconst_mask(mask.size(), CV_8U, cv::Scalar(0));
//    std::vector<std::vector<cv::Point>> cp2(1);
//    for (int i = 0; i < iz.size(); i++)
//    {
//        cp2[0].push_back(cv::Point(iz[i].real(), iz[i].imag()));
//    }
//    std::cout << cp2[0].size() << std::endl;
//    cv::drawContours(reconst_mask, cp2, 0, 1, 1);
//    cv::imshow("Reconstruct K < N", reconst_mask == 1);
//    cv::waitKey();


//    std::cout << "[";
//    for (int i = 0; i < U.total(); i++)
//    {
//        std::string sign;
//        float absi = std::fabs(U(i)[1]);
//        if (absi >= 0)
//            sign = " + ";
//        else
//            sign = " - ";

//        std::cout << U(i)[0] << sign << absi << "i";

//        if (i != U.total()-1)
//            std::cout << ";" << std::endl;
//        else
//            std::cout << "]" << std::endl;

//    }

    cv::dft(U, Fc); // apply DFT -> cartesian representation

    // calc polar representation
    cv::Mat_<float> planes[2];
    cv::split(Fc, planes);
    cv::cartToPolar(planes[0], planes[1], planes[0], planes[1]);
    cv::merge(planes, 2, Fp);
}

float FourierDescriptor::match(const FourierDescriptor& fd2)
{
    if (Fc.size() != fd2.Fc.size())
    {
        std::cout << "Descriptor sizes are not equal" << std::endl;
        return -1;
    }

    /*
     * translation invariance: F[0] := 0
     *     F[0] contains all translation related information
     * scale invariance: F[i] := F[i] / |F[1]|
     *     just scaling differences: F1[i] / |F2[1]|  =  F2[i] / F2[1]|)
     * rotation invariance: consider only |F[i]|
     *     rotation and starting point changes affect only the phase
     */

    float result = 0;
    for (int i = 2; i < /*10*/Fc.total(); i++)
    {
        float Fn1 =     Fp(i)[0] /     Fp(1)[0];
        float Fn2 = fd2.Fp(i)[0] / fd2.Fp(1)[0];
        float diff = Fn1 - Fn2;
//        std::cout << Fn1 << " - " << Fn2 << " = " << diff << std::endl;
        if (diff > 1)
            result += 1;
        else
            result += diff*diff;
    }
    return result;
}

cv::Mat_<cv::Vec2f> FourierDescriptor::reconstruct()
{
    cv::Mat_<cv::Vec2f> iU;
//    for (int i = num_samples/2; i < num_samples; i++)
//    {
//        Fc(i)[0] = 0;
//        Fc(i)[1] = 0;
//    }
    cv::dft(Fc, iU, cv::DFT_INVERSE | cv::DFT_SCALE);
    return iU;
}

void FourierDescriptor::dft(const int N, const int K)
{
    c = std::vector<std::complex<float>>(K, std::complex<float>(0, 0));
    for (int k = 0; k < K; k++)
    {
        for (int i = 0; i < N; i++)
        {
            float re =  std::cos(2.f * CV_PI * i * k / N);
            float im = -std::sin(2.f * CV_PI * i * k / N);
            std::complex<float> f(re, im);
            c[k] += z[i] * f;
        }
    }
}

void FourierDescriptor::idft(const int N, const int K)
{
    iz = std::vector<std::complex<float>>(N, std::complex<float>(0, 0));
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float re =  std::cos((-2.f * CV_PI * i * k) / N);
            float im = -std::sin((-2.f * CV_PI * i * k) / N);
            std::complex<float> f(re, im);
            iz[i] += f * c[k];
        }
        iz[i] *= 1.f/N;
    }
}

bool FourierDescriptor::less(cv::Point a, cv::Point b)
{
    if (a.x - center.x >= 0 && b.x - center.x < 0)
        return true;
    if (a.x - center.x < 0 && b.x - center.x >= 0)
        return false;
    if (a.x - center.x == 0 && b.x - center.x == 0) {
        if (a.y - center.y >= 0 || b.y - center.y >= 0)
            return a.y > b.y;
        return b.y > a.y;
    }

    // compute the cross product of vectors (center -> a) x (center -> b)
    int det = (a.x - center.x) * (b.y - center.y) -
              (b.x - center.x) * (a.y - center.y);
    if (det < 0)
        return true;
    if (det > 0)
        return false;

    // points a and b are on the same line from the center
    // check which point is closer to the center
    int d1 = (a.x - center.x) * (a.x - center.x) +
             (a.y - center.y) * (a.y - center.y);
    int d2 = (b.x - center.x) * (b.x - center.x) +
             (b.y - center.y) * (b.y - center.y);
    return d1 > d2;
}

void FourierDescriptor::sort()
{
    // estimate starting point (top center)
    int x0 = center.x, y0 = INT32_MAX;
    for (int y = 0; y < outline_mask.rows; y++)
    {
        if (outline_mask(y, x0) == 1 && y < y0)
            y0 = y;
    }

    cp.clear();
    cv::Mat_<uchar> marked = cv::Mat_<uchar>::zeros(outline_mask.size());
    int x = x0, y = y0; // current point
    int xn, yn; // potential successors
    int neighbor = 1; // [6 1 4]  // direct neighbors first starting right
                      // [3 x 0]  // then diagonal neighbors starting right
                      // [7 2 5]  // top before bottom

    cp.push_back(cv::Point(x0, y0)); // add starting point
    std::vector<cv::Point> n; // unmarked neighbors
    do {
        marked(y, x) = 1;
        n.clear();
        for (int i = 0; i < 8; i++)
        {
            switch (i) {
                case 0: // right
                    xn = x+1;
                    yn = y;
                    break;
                case 1: // top
                    xn = x;
                    yn = y-1;
                    break;
                case 2: // bottom
                    xn = x;
                    yn = y+1;
                    break;
                case 3: // left
                    xn = x-1;
                    yn = y;
                    break;
                case 4: // top right
                    xn = x+1;
                    yn = y-1;
                    break;
                case 5: // bottom right
                    xn = x+1;
                    yn = y+1;
                    break;
                case 6: // top left
                    xn = x-1;
                    yn = y-1;
                    break;
                case 7:
                    xn = x-1;
                    yn = y+1;
                    break;
                default:
                    std::cerr << "A point has only 8 neighbors" << std::endl;
                    break;
            }
            if (outline_mask(yn, xn) == 1 && marked(yn, xn) == 0)
            {
                n.push_back(cv::Point(xn, yn));
            }
        }


        // sort neighbors
        std::sort(n.begin(), n.end(), [this](cv::Point a, cv::Point b) {
            return less(a, b);
        });

        if (n.size() > 0)
        {
            cp.push_back(n[0]); // add successor
            x = n[0].x;
            y = n[0].y;
//            std::cout << n[0] << " - " << ((uint)marked(n[0].y, n[0].x)) << std::endl;
        }

        // returned to start point or no unmarked neighbor
    } while ((x != x0 || y != y0) && n.size() > 0);
}
