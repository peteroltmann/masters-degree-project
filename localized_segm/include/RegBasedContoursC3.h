#ifndef LOCALIZED_CONTOURS_H
#define LOCALIZED_CONTOURS_H

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <exception>
#include <list>

#define WINDOW "Contour Evolution"
#define CHAN_VESE 0
#define YEZZI 1

/*!
 * \brief Region-based contours class.
 *
 * This class offers the WINDOWality to apply different region-based active
 * contours using the level-set method (narrow band).
 */
class RegBasedContoursC3
{
public:
    RegBasedContoursC3(); //!< The default constructor.
    virtual ~RegBasedContoursC3(); //!< The default destructor.

    /*!
     * \brief Apply a region-based active contour algorithm to the specified
     * image using the sparse-field method.
     *
     * \param frame         the image to apply the algorithm to
     * \param initMask      the initialization mask for the level-set WINDOW
     * \param phi           the level-set WINDOW (empty cv::Mat)
     * \param iterations    the number of iterations
     * \param method        the contour's speed WINDOW method
     * \param localized     weather the localized version of the specified
     *                      method is supposed to be used
     * \param rad           the radius of localized regions
     * \param alpha         the curvature weight (higher -> smoother)
     */
    void applySFM(cv::Mat& frame, cv::Mat initMask, int iterations,
                  int method=0, bool localized=false, int rad=18,
                  float alpha=.2f,
                  cv::Vec3f a=cv::Vec3f(1.f/3.f, 1.f/3.f, 1.f/3.f));

    /*!
     * \brief Create a signed distance WINDOW (SDF) from a mask.
     *
     * \param mask  the mask that describes the initial contour: 1 inside and 0
     *              outside the contour.
     * \return      the signed distance WINDOW (SDF)
     */
    cv::Mat mask2phi(cv::Mat mask);

    /*!
     * \brief Set the frame for contour evolution.
     * \param frame the frame to be set
     */
    void setFrame(cv::Mat& frame);

    /*!
     * \brief Initialize the sparse-field method.
     * \param initMask  the initialization mask
     */
    void init(cv::Mat &initMask);

    /*!
     * Do one iteration step of the sparse-field method.
     */
    void iterate();

    /*!
     * \brief Calculate the force affecting the contour.
     */
    void calcF();

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
    bool pushBack(int listNo, bool tmp, cv::Point p, cv::Size size);

    bool _localized;
    int _method;
    float _alpha;
    float _rad;

    cv::Vec3f _a; //!< The weighting factors for each channel

    cv::Vec3f _sumInt;
    cv::Vec3f _sumExt;
    float _cntInt;
    float _cntExt;
    cv::Vec3f _meanInt;
    cv::Vec3f _meanExt;

public:
    cv::Mat _image;   //!< Current original frame
    cv::Mat _frame; //!< Current frane in CV_32F
    cv::Mat _phi;
    cv::Mat _label;
    cv::Mat _F;
    std::list<cv::Point> _lz, _ln1, _lp1, _ln2, _lp2; //!< Level set lists.
    std::list<cv::Point> _sz, _sn1, _sp1, _sn2, _sp2; //!< Temporary lists.
    std::list<cv::Point>::iterator _lz_it, _ln1_it, _lp1_it, _ln2_it, _lp2_it;
    std::list<cv::Point>::iterator _sz_it, _sn1_it, _sp1_it, _sn2_it, _sp2_it;
};

#endif // LOCALIZED_CONTOURS_H
