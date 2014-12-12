#ifndef CONTOUR_H
#define CONTOUR_H

#include <opencv2/core/core.hpp>

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
     * \brief Transform the contour mask with the specified affine parameters.
     * \param state the estimated state of the particle filter including the
     *              affine parameters
     */
    void transform_affine(cv::Mat_<float>& state);

    /*!
     * \brief Evolve the contour.
     * \param segm          the segmentation to use
     * \param frame         the current frame
     * \param iterations    the number of iterations
     */
    void evolve_contour(RegBasedContours& segm, cv::Mat &frame, int iterations);

    /*!
     * \brief Calculate the contour energy.
     * \param   segm    Contour evolution object reference, that holds the level
     *                  set data.
     */
    void calc_energy(cv::Mat& frame);

    /*!
     * \brief Calculate the distance of two contours. The level set data of the
     * other contour (<tt>phi_mu</tt>) has to be saved in the used intance of
     * this class.
     *
     * \param   segm    Contour evolution object reference, that holds the level
     *                  set data.
     */
    void calc_distance(RegBasedContours& segm);

    cv::Mat_<uchar> contour_mask; //!< Masks interior pixel (interior = 1/ != 0)
    cv::Mat_<float> phi_mu; //!< Level set function before contour evolution.
    float energy; //!< the calculated contour energy

    /*!
     * the calculated contour distance from before to after
     */
    float distance;

};

#endif // CONTOUR_H
