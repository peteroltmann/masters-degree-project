#include "hist.h"

#include <opencv2/core/core.hpp>
#include <iostream>

void calc_hist(cv::Mat& bgr, cv::Mat& hist, cv::Mat& mask)
{
    static const int channels[] = {0};
    static const int hist_size[] = {256};
    static const float range[] = {0, 255};
    static const float* ranges[] = {range};
//    static const cv::Mat mask;
    static const int dims = 1;
    cv::Mat srcs[] = {bgr};

    calcHist(srcs, sizeof(srcs), channels, mask, hist, dims, hist_size, ranges,
             true, false);
}


float match_shapes(cv::Mat_<uchar>& shape_mask_1, cv::Mat_<uchar>& shape_mask_2,
                   int method)
{
    double ma[7];
    double mb[7];

    cv::Moments m1 = cv::moments(shape_mask_1, true);
    cv::Moments m2 = cv::moments(shape_mask_2, true);
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
                std::cerr << "Unknown comparison methods" << std::endl;
                break;
            }

            result += fabs( ama - amb );
        }
    }

    return result;
}


cv::Mat draw_hist(cv::Mat& hist)
{
    cv::Mat hist_draw;
    hist.copyTo(hist_draw);

    int histSize = 256;
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    cv::Mat histImage(hist_h, hist_w, CV_8U, cv::Scalar(0,0,0));

    cv::normalize(hist_draw, hist_draw, 0, histImage.rows, 32, -1, cv::Mat() );

    for(int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist_draw.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), hist_h - cvRound(hist_draw.at<float>(i)) ),
                            cv::Scalar( 255, 0, 0), 2, 8, 0  );
    }

    return histImage;
}

float calcBC(cv::Mat_<float>& hist1, cv::Mat_<float>& hist2)
{
    if (hist1.type() != CV_32F && hist2.type() != CV_32F)
    {
        std::cout << "Histogram types != CV_32F" << std::endl;
        return -1;
    }

    if (hist1.rows != hist2.rows && hist1.cols == 1 && hist2.cols == 1)
    {
        std::cout << "Histogram sizes are not equal" << std::endl;
        return -1;
    }

    float bc = 0;
    float* hist1p = hist1.ptr<float>();
    float* hist2p = hist2.ptr<float>();
    for (int x = 0; x < hist1.total(); x++)
    {
        bc += std::sqrt(hist1p[x] * hist2p[x]);
    }

    return 1 - bc; // 0 = perfect match, 1 = total missmatch
}


void normalize(cv::Mat_<float>& hist)
{
    float sum = 0.f;
    float* hist_p = hist.ptr<float>();
    for (int x = 0; x < hist.total(); x++)
    {
        sum += hist_p[x];
    }

    for (int x = 0; x < hist.total(); x++)
    {
        hist_p[x] /= sum;
    }
}


cv::Rect bounding_rect(cv::Mat_<uchar>& mask)
{
    // get contour points from mask
    std::vector<cv::Point> contour_points;
    for (int y = 0; y < mask.rows; y++)
    {
        uchar* cm = mask.ptr(y);
        for (int x = 0; x < mask.cols; x++)
        {
            if (cm[x] == 1)
                contour_points.push_back(cv::Point(x, y));
        }
    }

    return cv::boundingRect(contour_points);
}
