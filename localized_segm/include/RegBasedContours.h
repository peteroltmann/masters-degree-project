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
 * This class offers the functionality to apply different region-based active
 * contours using the level-set method (narrow band).
 */
class RegBasedContours
{
public:
    RegBasedContours(); //!< The default constructor.
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
    void applySFM(cv::Mat& frame, cv::Mat initMask, int iterations,
                  int method=0, bool localized=false, int rad=18,
                  float alpha=.2f);

    /*!
     * \brief Apply a region-based active contour algorithm to the specified
     * image.
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
    void apply(cv::Mat frame, cv::Mat initMask, cv::Mat& phi, int iterations,
               int method=1, bool localized=false, int rad=18, float alpha=.2f);

    /*!
     * \brief Sussman-reinitialization to retain the level-set function to b a
     * signed distance function (SDF).
     *
     * \param D     the function to apply the Sussman-reinitialization to
     * \param dt    the time step.
     */
    void sussmanReinit(cv::Mat&D, float dt);

    /*!
     * \brief Create a signed distance function (SDF) from a mask.
     *
     * \param mask  the mask that describes the initial contour: 1 inside and 0
     *              outside the contour.
     * \return      the signed distance function (SDF)
     */
    cv::Mat mask2phi(cv::Mat mask);

    void setFrame(cv::Mat& frame);
    void init(cv::Mat &initMask);
    void iterate();
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

    float _sumInt;
    float _sumExt;
    float _cntInt;
    float _cntExt;
    float _meanInt;
    float _meanExt;

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
