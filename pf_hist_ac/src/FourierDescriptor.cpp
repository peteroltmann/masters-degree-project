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
    U = cv::Mat_<cv::Vec2f>(num_samples, 1);

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
//    std::reverse(cp.begin(), cp.end());
//    std::sort(cp.begin(), cp.end(), [this](cv::Point a, cv::Point b) {
//        return less(a, b);
//    });

//    cv::Mat asd;
//    outline_mask.copyTo(asd);
//    for (int i = 0, j = 0; i < cp.size(); i+=cp.size()/num_samples, j++)
//    {
////        std::stringstream ss;
////        ss << j;
////        std::cout << ss.str() << std::endl;
////        cv::putText(asd, ss.str(), cp[i], 1, .5, 255);
//        cv::circle(asd, cp[i], 0, 255);
//        cv::imshow("YO2", asd);
//        cv::waitKey();
//    }

    // create complex vector
    int num_points = cp.size();
    int delta = num_points / num_samples;
    for (int i = 0, j = 0; i < num_points && j < num_samples; i += delta, j++)
    {
        U(j) = cv::Vec2f(cp[i].x, cp[i].y);
    }

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
        std::cout << "Histogram sizes are not equal" << std::endl;
        return EXIT_FAILURE;
    }

    /*
     * translation invariance: F[0] := 0
     *     F[0] contains all translation related information
     * scale invariance: F[i] := F[i] / |F[1]|
     *     just scaling differences: F1[i] / |F2[1]|  =  F2[i] / F2[1]|)
     * rotation invariance: consider only |F[i]|
     *     rotation and starting point changes affect only the phase
     */

    float sum = 0;
    for (int i = 2; i < /*10*/Fc.total(); i++)
    {
        float Fn1 =     Fp(i)[0] /     Fp(1)[0];
        float Fn2 = fd2.Fp(i)[0] / fd2.Fp(1)[0];
        float diff = Fn1 - Fn2;
//        std::cout << Fn1 << " - " << Fn2 << " = " << diff << std::endl;
        if (diff > 1)
            sum += 1;
        else
            sum += diff*diff;
    }
    return sum;
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
