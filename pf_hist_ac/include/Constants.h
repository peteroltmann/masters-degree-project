#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <opencv2/core/core.hpp>

// colors
const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);

// window names
const std::string WINDOW_NAME("ObjectTracker");
const std::string WINDOW_DETAILED_NAME("Detailed Result");
const std::string WINDOW_TEMPALTE_NAME("Template");
const std::string WINDOW_CV_NAME("Characteristic Views");

#endif // CONSTANTS_H
