#ifndef LOCALIZED_CONTOURS_H
#define LOCALIZED_CONTOURS_H

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <exception>

#define WINDOW_NAME "Image"
#define CHAN_VESE 0
#define YEZZI 1

class RegBasedContours
{
public:
    RegBasedContours();
    virtual ~RegBasedContours();

    void apply(cv::Mat frame, cv::Mat initMask, cv::Mat& seg, int iterations,
               int method=1, bool localized=false, int rad=18, float alpha=.2f);
    void sussmanReinit(cv::Mat&D, float dt);

private:
    cv::Mat mask2phi(cv::Mat mask);
};

/*!
 * \brief   Compute max value of a cv::Mat. Must not be empty - no check!
 * \param   mat the mat whose max value shoud be computed
 * \return  the max value
 * \throws  runtime_error on passing an empty cv::Mat
 */
template <typename T>
T max(cv::Mat& mat)
{
    if (mat.empty())
        throw std::runtime_error("Empty cv::Mat object.");

    T* ptr = mat.ptr<T>();
    T maxVal = ptr[0];
    for (int i = 1; i < mat.rows*mat.cols; i++)
        if (ptr[i] > maxVal)
            maxVal = ptr[i];
    return maxVal;
}

/*!
 * \brief   Compute min value of a cv::Mat. Must not be empty - no check!
 * \param   min the mat whose max value shoud be computed
 * \return  the min value
 * \throws  runtime_error on passing an empty cv::Mat
 */

template <typename T>
T min(cv::Mat& mat)
{
    if (mat.empty())
        throw std::runtime_error("Empty cv::Mat object.");

    T* ptr = mat.ptr<T>();
    T minVal = ptr[0];
    for (int i = 1; i < mat.rows*mat.cols; i++)
        if (ptr[i] < minVal)
            minVal = ptr[i];
    return minVal;
}

#endif // LOCALIZED_CONTOURS_H
