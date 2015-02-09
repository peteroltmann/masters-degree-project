#include "Histogram.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

Histogram::Histogram() {}

Histogram::Histogram(const Histogram& other)
{
    other.data.copyTo(data);
}

Histogram&Histogram::operator=(const Histogram& other)
{
    other.data.copyTo(data);
}

Histogram::~Histogram() {}

void Histogram::calc_hist(cv::Mat& img, int type, const cv::Mat& mask)
{
    if (type == GRAY)
        calc_hist_1(img, mask);
    else
        calc_hist_3(img, mask);

    normalize();
}

void Histogram::adapt(const Histogram& hist2, float a)
{
    if (data.type() != hist2.data.type())
    {
        throw cv::Exception(-1, "Histogram types are not equal",
                            "Histogram::match", "Histogram.cpp", 0);
    }

    if (data.total() != hist2.data.total())
    {
        throw cv::Exception(-1, "Histogram sizes are not equal",
                            "Histogram::match()", "Histogram.cpp", 0);
    }

    data = (1-a) * data + a * hist2.data;
    normalize();
}

void Histogram::calc_hist_1(cv::Mat& img, const cv::Mat& mask)
{
    static const int channels[] = {0};
    static const int hist_size[] = {256};
    static const float range[] = {0, 256};
    static const float* ranges[] = {range};
    static const int dims = 1;
    cv::Mat srcs[] = {img};

    cv::calcHist(srcs, sizeof(srcs), channels, mask, data, dims, hist_size,
                 ranges, true, false);
}

void Histogram::calc_hist_3(cv::Mat& img, const cv::Mat& mask)
{
    static const int channels[] = {0, 1, 2};
    static const int b_bins = 16;
    static const int g_bins = 16;
    static const int r_bins = 16;
    static const int hist_size[] = {b_bins, g_bins, r_bins};
    static const float branges[] = {0, 256};
    static const float granges[] = {0, 256};
    static const float rranges[] = {0, 256};
    static const float* ranges[] = {branges, granges, rranges};
    static const int dims = 3;
    cv::Mat srcs[] = {img};

    cv::calcHist(srcs, sizeof(srcs), channels, mask, data, dims, hist_size,
                 ranges, true, false);
}

void Histogram::normalize()
{
    float sum = 0.f;
    float* data_ptr = data.ptr<float>();
    for (int x = 0; x < data.total(); x++)
    {
        sum += data_ptr[x];
    }

    for (int x = 0; x < data.total(); x++)
    {
        data_ptr[x] /= sum;
    }
}

float Histogram::match(Histogram& hist2)
{
    if (data.type() != CV_32F && hist2.data.type() != CV_32F)
    {
        throw cv::Exception(-1, "Histogram types != CV_32F", "Histogram::match",
                            "Histogram.cpp", 0);
    }

    if (data.total() != hist2.data.total())
    {
        throw cv::Exception(-1, "Histogram sizes are not equal",
                            "Histogram::match()", "Histogram.cpp", 0);
    }

    float bc = 0;
    float* data_ptr = data.ptr<float>();
    float* data2_ptr = hist2.data.ptr<float>();
    for (int x = 0; x < data.total(); x++)
    {
        bc += std::sqrt(data_ptr[x] * data2_ptr[x]);
    }

    return 1 - bc; // 0 = perfect match, 1 = total missmatch
}


cv::Mat Histogram::draw(const cv::Mat& frame, int type, const cv::Mat& mask)
{
    if (type == GRAY)
        return draw_1();
    else
        return draw_3(frame, type, mask);
}

cv::Mat Histogram::draw_1()
{
    static const int hist_size = 256;
    static const int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((float)hist_w/hist_size);

    cv::Mat_<float> hist_draw;
    cv::Mat histImage(hist_h, hist_w, CV_8U, cv::Scalar(0));

    cv::normalize(data, hist_draw, 0, histImage.rows, 32, -1, cv::Mat());

    for(int i = 1; i < hist_size; i++)
    {
        cv::Point pt1 = cv::Point(bin_w*(i-1),
                                  hist_h - cvRound(hist_draw(i-1)));
        cv::Point pt2 = cv::Point(bin_w*(i),
                                  hist_h - cvRound(hist_draw(i)));
        cv::line(histImage, pt1, pt2, cv::Scalar(255), 2, 8, 0);
    }

    return histImage;
}

cv::Mat Histogram::draw_3(const cv::Mat& frame, int type, const cv::Mat& mask)
{
    // RGB
    cv::Scalar color1 = cv::Scalar(0, 0, 255);
    cv::Scalar color2 = cv::Scalar(0, 255, 0);
    cv::Scalar color3 = cv::Scalar(255, 0, 0);

    if (type == BGR)
    {
        color1 = cv::Scalar(255, 0, 0);
        color2 = cv::Scalar(0, 255, 0);
        color3 = cv::Scalar(0, 0, 255);
    }
    else if (type == HSV)
    {
        color1 = cv::Scalar(0, 0, 255);
        color2 = cv::Scalar(0, 255, 0);
        color3 = cv::Scalar(255, 0, 0);
    }

    std::vector<cv::Mat> rgb_planes;
    cv::split(frame, rgb_planes);

    static const int hist_size = 256;
    static const float range[] = {0, 256};
    static const float* ranges[] = {range};
    static const int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((float)hist_w/hist_size);

    cv::Mat_<float> r_hist, g_hist, b_hist;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::calcHist(&rgb_planes[0], 1, 0, mask, r_hist, 1, &hist_size, ranges,
                 true, false);
    cv::calcHist(&rgb_planes[1], 1, 0, mask, g_hist, 1, &hist_size, ranges,
            true, false);
    cv::calcHist(&rgb_planes[2], 1, 0, mask, b_hist, 1, &hist_size, ranges,
            true, false);

    cv::normalize(r_hist, r_hist, 0, histImage.rows, 32, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows, 32, -1, cv::Mat());
    cv::normalize(b_hist, b_hist, 0, histImage.rows, 32, -1, cv::Mat());

    for(int i = 1; i < hist_size; i++)
    {
        cv::line(histImage,
                 cv::Point(bin_w*(i-1), hist_h - cvRound(r_hist(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(r_hist(i))),
                 color1, 2, 8, 0);
        cv::line(histImage,
                 cv::Point(bin_w*(i-1), hist_h - cvRound(g_hist(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(g_hist(i))),
                 color2, 2, 8, 0);
        cv::line(histImage,
                 cv::Point(bin_w*(i-1), hist_h - cvRound(b_hist(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(b_hist(i))),
                 color3, 2, 8, 0);
    }

    return histImage;
}

bool Histogram::empty()
{
    return data.empty();
}
