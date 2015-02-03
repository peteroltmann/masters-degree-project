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

    /*!
     * \brief Match the contour shape to a given other contour shape using
     *        the Hu-Moments.
     *
     * Re-implementaion of the OpenCV-function <tt>matchShapes()</tt> due to a
     * bug in version 2.4.9 that causes the function to always return zero.
     *
     * \param contour2  the contour to be matched with.
     * \param method    comparison method: <tt>CV_CONTOURS_MATCH_I1</tt> ,
     *                  <tt>CV_CONTOURS_MATCH_I2</tt> or
     *                  <tt>CV_CONTOURS_MATCH_I3</tt>
     *
     * TODO:
     * Only CV_CONTOURS_MATCH_I1 and 2 supported right now. Experimentally
     * method "4"
     *
     * See details of the comparison method:
     * http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#matchshapes
     * \return
     */
    float match(Contour& contour2, int method=CV_CONTOURS_MATCH_I1);

    /*!
     * \brief Draw the contour.
     * \param window_image  the output image
     * \param color         the color
     */
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

    //! Return weather the contour mask ist empty or not.
    bool empty();

    cv::Mat_<uchar> mask; //!< Masks interior pixel (interior = 1/ != 0)
    cv::Rect bound;       /*!< Bounding rectangle of the contour. Set with
                               bounding_rect(). */
    cv::Mat_<uchar> roi;  /*!< Contour ROI. Recommended to be set with
                               bounding_rect(). */
};

#endif // CONTOUR_H
