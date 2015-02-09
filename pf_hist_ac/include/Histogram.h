#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/core/affine.hpp>

#define GRAY 0
#define RGB  1
#define BGR  2
#define HSV  3

/*!
 * \brief The Histogram class.
 */
class Histogram
{
public:
    Histogram(); //!< The default constructor.
    Histogram(const Histogram& other); //!< The copy constructor.
    Histogram& operator=(const Histogram& other); //!< The assignement operator.
    virtual ~Histogram(); //!< The default destructor.

    /*!
     * \brief Calculate the histogram of the given image in the specified color
     * space.
     *
     * \param img   the image whose histogram is to be calculated.
     * \param type  color space (GRAY, RGB, BGR, HSV)
     * \param mask  optional mask
     */
    void calc_hist(cv::Mat& img, int type, const cv::Mat& mask=cv::Mat());

private:
    //! Calculate one chanel histogram.
    void calc_hist_1(cv::Mat& img, const cv::Mat& mask);

    //! Calculate three chanel histogram.
    void calc_hist_3(cv::Mat& img, const cv::Mat& mask);

public:
    //! Normalize the histogram data.
    void normalize();

    /*!
     * \brief Match histograms using the Bhattacharyya coefficient (BC).
     *
     * The return value (1 - BC) can be interpreted as distance between two
     * histograms.
     *
     * \param hist2 the histogram to be compared with
     * \return 1 - BC
     */
    float match(Histogram& hist2);

    /*!
     * \brief Draw histogram with specified type.
     * \param type  color space (GRAY, RGB, BGR, HSV)
     * \param frame current frame when using mulity-channel type because the
     *              histogram has to be (re-)calculated channel-wise
     * \param mask  optional mask for histogram (re-)calculation
     * \return the drawn histogram image
     */
    cv::Mat draw(const cv::Mat& frame=cv::Mat(), int type=GRAY,
                 const cv::Mat& mask=cv::Mat());
private:
    cv::Mat draw_1(); //!< Draw one chanel histogram.

    /*!
     * \brief Draw three chanel histogram (split channels).
     * \param frame current frame because the histogram has to be
     *              (re-)calculated channel-wise
     * \param mask  optional mask for histogram (re-)calculation
     * \param type  color space (RGB, BGR, HSV)
     * \return the drawn histogram image
     */
    cv::Mat draw_3(const cv::Mat& frame, int type,
                   const cv::Mat& mask=cv::Mat());

public:
    //! Return weather the histogram data ist empty or not.
    bool empty();

private:
    cv::Mat_<float> data; //!< The histogram data.

};

#endif // HISTOGRAM_H
