#ifndef HIST_H
#define HIST_H

#include <opencv2/imgproc/imgproc.hpp>

// TODO: make "helper" functions module from this

void calc_hist(cv::Mat& bgr, cv::Mat& hist, cv::Mat mask=cv::Mat());

cv::Mat draw_hist(cv::Mat& hist);

/*!
 * \brief Match shapes using the Hu-Moments.
 *
 * Re-implementaion of the OpenCV-function <tt>matchShapes()</tt> due to a bug
 * in version 2.4.9 that causes the function to always return zero.
 *
 * \param shape_mask_1  binary image of the first shape to match
 * \param shape_mask_2  binary image of the second shape to match
 * \param method        comparison method: <tt>CV_CONTOURS_MATCH_I1</tt> ,
 *                      <tt>CV_CONTOURS_MATCH_I2</tt> or
 *                      <tt>CV_CONTOURS_MATCH_I3</tt>
 *
 * Only CV_CONTOURS_MATCH_I1 supported right now.
 *
 * See details of the comparison method:
 * http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#matchshapes
 */
float match_shapes(cv::Mat_<uchar>& shape_mask_1, cv::Mat_<uchar>& shape_mask_2,
                  int method=CV_CONTOURS_MATCH_I1);

float calcBC(cv::Mat_<float>& hist1, cv::Mat_<float>& hist2);

void normalize(cv::Mat_<float>& hist);

cv::Rect bounding_rect(cv::Mat_<uchar>& mask);

#endif // HIST_H
