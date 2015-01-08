#ifndef CONTOUR_H
#define CONTOUR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class RegBasedContours;

/*!
 * \brief The Contour class to represent an active contour for tracking
 * via level set particle filters.
 */
class Contour
{
public:
    Contour(); //!< The default constructor.
    virtual ~Contour(); //!< The default destructor.

    /*!
     * \brief Construct a contour object with a given mask.
     * \param mask  the contour mask
     */
    Contour(const cv::Mat_<uchar>& mask);

    /*!
     * \brief Transform the contour mask with the specified affine parameters.
     * \param state the estimated state of the particle filter including the
     *              affine parameters
     */
    void transform_affine(cv::Mat_<float>& state);

    /*!
     * \brief Evolve the contour.
     * \param segm          the segmentation object to use
     * \param frame         the current frame
     * \param iterations    the number of iterations
     */
    void evolve(RegBasedContours& segm, cv::Mat& frame, int iterations);

    /*!
     * \brief Calculate bounding rectangle of the contour.
     * \return the bounding rectangle
     */
    cv::Rect bounding_rect();

    float match(Contour& contour2, int method=CV_CONTOURS_MATCH_I1);

    void draw(cv::Mat& window_image, cv::Scalar color);

    /*!
     * \brief Set the contour mask by copyiing it.
     * \param mask the contour mask to be set
     */
    void set_mask(const cv::Mat& mask);

    /*!
     * \brief Set the contour ROI.
     * \param rect  the rectangle describing the ROI. Reommended to be set with
     *              bounding_rect().
     */
    void set_roi(cv::Rect rect);

    cv::Mat_<uchar> mask; //!< Masks interior pixel (interior = 1/ != 0)
    cv::Mat_<uchar> roi;  /*!< Contour ROI. Recommended to be set with
                               bounding_rect(). */
};

#endif // CONTOUR_H
