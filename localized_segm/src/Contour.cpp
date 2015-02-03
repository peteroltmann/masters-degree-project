#include "Contour.h"
#include "RegBasedContours.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

Contour::Contour() {}

Contour::~Contour() {}

Contour::Contour(const cv::Mat_<uchar>& mask)
{
    set_mask(mask);
}

void Contour::evolve(RegBasedContours& segm, cv::Mat& frame, int iterations)
{
    segm.setFrame(frame);
    segm.init(mask);

    for (int its = 0; its < iterations; its++)
    {
        segm.iterate();
    }

    // get rid of eventual blobs
    cv::Mat inOut = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
    inOut.setTo(255, segm._phi <= 0);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // acutalize mask
    mask.setTo(0);
    cv::drawContours(mask, contours, -1, 1, CV_FILLED); // set to 1 (as color)
    set_roi(bounding_rect());
}

cv::Rect Contour::bounding_rect()
{
    // get contour points from mask
    std::vector<cv::Point> contour_points;
    for (int y = 0; y < mask.rows; y++)
    {
        const uchar* cm = mask.ptr(y);
        for (int x = 0; x < mask.cols; x++)
        {
            if (cm[x] == 1)
                contour_points.push_back(cv::Point(x, y));
        }
    }

    return cv::boundingRect(contour_points);
}

float Contour::match(Contour& contour2, int method)
{
    double ma[7];
    double mb[7];

    cv::Moments m1 = cv::moments(roi, true);
    cv::Moments m2 = cv::moments(contour2.roi, true);
    cv::HuMoments(m1, ma);
    cv::HuMoments(m2, mb);

    int sma, smb;
    float eps = 1.e-5;
    float result = 0.f;
    for (int i = 0; i < 7; i++)
    {
        float ama = fabs( ma[i] );
        float amb = fabs( mb[i] );

        if( ma[i] > 0 )
            sma = 1;
        else if( ma[i] < 0 )
            sma = -1;
        else
            sma = 0;
        if( mb[i] > 0 )
            smb = 1;
        else if( mb[i] < 0 )
            smb = -1;
        else
            smb = 0;

        if( ama > eps && amb > eps )
        {
            switch(method)
            {
                case CV_CONTOURS_MATCH_I1:
                    ama = 1. / (sma * log10(ama));
                    amb = 1. / (smb * log10(amb));
                    break;
                case CV_CONTOURS_MATCH_I2:
                    ama = sma * log10(ama);
                    amb = smb * log10(amb);
                    break;
                case 4:
                    ama = sma * ama;
                    amb = smb * amb;
                    break;
                default:
                    result = -1;
            }

            if (result == -1)
            {
                std::cerr << "Unknown comparison method" << std::endl;
                break;
            }

//            result += fabs(ama - amb); // TODO: abs? log?
            result += (ama - amb)*(ama - amb);
        }
    }

    return result;
}

void Contour::draw(cv::Mat& window_image, cv::Scalar color)
{
    if (window_image.size() != mask.size())
    {
        std::cerr << "Error drawing contour: " << "window_image.size() "
                  << window_image.size() << " != mask.size() "
                  << mask.size() << std::endl;
        return;
    }

    cv::Mat inOut = cv::Mat::zeros(window_image.rows, window_image.cols, CV_8U);
    inOut.setTo(255, mask == 1);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(inOut, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(window_image, contours, -1, color, 1);
}

void Contour::set_mask(const cv::Mat& mask)
{
    mask.copyTo(this->mask);
    set_roi(bounding_rect());
}

void Contour::set_roi(cv::Rect rect)
{
    bound = rect;
    roi = cv::Mat(mask, rect);
}

bool Contour::empty()
{
    return mask.empty();
}
