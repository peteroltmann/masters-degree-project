#ifndef LOCALIZED_CONTOURS_H
#define LOCALIZED_CONTOURS_H

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <exception>
#include <list>

#define WINDOW "Contour Evolution"

/*!
 * \brief The contour's speed function method.
 */
enum Method
{
    CHAN_VESE = 0,
    YEZZI = 1
};

/*!
 * \brief Region-based contours class.
 *
 * This class offers the functionality to apply different region-based active
 * contours using the level-set method (narrow band).
 */
class RegBasedContours
{
public:
    /*!
     * \brief RegBasedContours
     * \param method        the contour's speed function method
     * \param localized     weather the localized version of the specified
     *                      method is supposed to be used
     * \param rad           the radius of localized regions
     * \param alpha         the curvature weight (higher -> smoother)
     */
    RegBasedContours(Method method=CHAN_VESE, bool localized=false, int rad=18,
                     float alpha=.2f);

    virtual ~RegBasedContours(); //!< The default destructor.

    /*!
     * \brief Apply a region-based active contour algorithm to the specified
     * image using the sparse-field method.
     *
     * \param frame         the image to apply the algorithm to
     * \param initMask      the initialization mask for the level-set function
     * \param phi           the level-set function (empty cv::Mat)
     * \param iterations    the number of iterations
     * \param method        the contour's speed function method
     * \param localized     weather the localized version of the specified
     *                      method is supposed to be used
     * \param rad           the radius of localized regions
     * \param alpha         the curvature weight (higher -> smoother)
     */
    void applySFM(cv::Mat& frame, cv::Mat init_mask, int iterations);

    /*!
     * \brief Apply a region-based active contour algorithm to the specified
     * image using the narrow-band method..
     *
     * \param frame         the image to apply the algorithm to
     * \param initMask      the initialization mask for the level-set function
     * \param phi           the level-set function (empty cv::Mat)
     * \param iterations    the number of iterations
     * \param method        the contour's speed function method
     * \param localized     weather the localized version of the specified
     *                      method is supposed to be used
     * \param rad           the radius of localized regions
     * \param alpha         the curvature weight (higher -> smoother)
     */
    void apply(cv::Mat frame, cv::Mat init_mask, int iterations);

    /*!
     * \brief Sussman-reinitialization to retain the level-set function to b a
     * signed distance function (SDF).
     *
     * \param D     the function to apply the Sussman-reinitialization to
     * \param dt    the time step.
     */
    void sussman_reinit(cv::Mat&D, float dt);

    /*!
     * \brief Create a signed distance function (SDF) from a mask.
     *
     * \param mask  the mask that describes the initial contour: 1 inside and 0
     *              outside the contour.
     * \return      the signed distance function (SDF)
     */
    cv::Mat mask2phi(cv::Mat mask);

    /*!
     * \brief Set the current frame.
     * \param frame the current frame
     */
    void set_frame(cv::Mat& frame);

    /*!
     * \brief Set parameters of the sparse-field method.
     *
     * \param method        the contour's speed function method
     * \param localized     weather the localized version of the specified
     *                      method is supposed to be used
     * \param rad           the radius of localized regions
     * \param alpha         the curvature weight (higher -> smoother)
     */
    void set_params(Method method, bool localized, int rad, float alpha);

    /*!
     * \brief Initialization for the sparse-field method.
     * \param initMask  initialization mask
     */
    void init(cv::Mat &initMask);

    /*!
     * \brief Do one iteration step of the sparse-field method.
     */
    void iterate();

    /*!
     * \brief Calculate the force F (sparse-field method).
     */
    void calc_F();

private:
    /*!
     * \brief Adds specified point to a specified (temporay) list with checking
     * image bounds.
     *
     * \param listNo    the number of the list
     * \param tmp       weather the point should be added to the temporary list
     * \param p         the point to be added
     * \param size      the image size for bound check
     * \return          weather the point was added
     */
    bool push_back(int listNo, bool tmp, cv::Point p, cv::Size size);

    Method method; //!< The energy used for contour evolution
    bool localized; //!< Use the localized version of the energy or not
    float rad; //<! Radius (only used in localized version)
    float alpha; //!< Factor to control the influence of the curvature

    float sum_int; //!< Sum of values inside the contour
    float sum_ext; //!< Sum of values outside the contour
    float cnt_int; //!< Number of points inside the contour
    float cnt_ext; //!< Number of points outside the contour
    float mean_int; //!< Mean value inside the contour
    float mean_ext; //!< Mean value outside the contour

    cv::Mat image; //!< Current original frame
    cv::Mat frame; //!< Current frame in CV_32F
    cv::Mat label; //!< The label map
    cv::Mat F; //! The force effecting the contour
    std::list<cv::Point> lz, ln1, lp1, ln2, lp2; //!< Level set lists
    std::list<cv::Point> sz, sn1, sp1, sn2, sp2; //!< Temporary lists
    std::list<cv::Point>::iterator lz_it, ln1_it, lp1_it, ln2_it, lp2_it;
    std::list<cv::Point>::iterator sz_it, sn1_it, sp1_it, sn2_it, sp2_it;

public:
    cv::Mat phi; //!< The surface function
};

#endif // LOCALIZED_CONTOURS_H
