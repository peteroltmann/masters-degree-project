#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <opencv2/core/core.hpp>

const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);

const std::string WINDOW_NAME("Image");
const std::string WINDOW_FRAME_NAME("Frame");
const std::string WINDOW_TEMPALTE_NAME("Template");
const std::string WINDOW_RECONSTR_NAME("Reconstructed");
const std::string WINDOW_RECONSTR_TEMPL_NAME("Reconstructed Template");

#endif // CONSTANTS_H
