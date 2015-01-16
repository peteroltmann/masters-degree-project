#ifndef FOURIER_DESCRIPTOR_H
#define FOURIER_DESCRIPTOR_H

#include <opencv2/core/core.hpp>

class FourierDescriptor
{
public:

    /*!
     * \brief Construct fourier descriptor from a given contour mask.
     * \param mask  masks interior pixel (interior = 1/ != 0)
     */
    FourierDescriptor(const cv::Mat_<uchar>& mask, int num_samples=128);

    virtual ~FourierDescriptor(); //!< The default destructor.

    /*!
     * \brief init  Initialize fourier descriptor from a given contour mask.
     * \param mask  masks interior pixel (interior = 1/ != 0)
     */
    void init(const cv::Mat_<uchar>& mask, int num_samples);

    /*!
     * \brief Normalize to obtain invariance.
     */
    float match(const FourierDescriptor& fd2);

//private:
    int num_samples; //!< number of sample points taken from the contour
    cv::Point center; //!< center of contour
    std::vector<cv::Point> cp; //!< contour points sorted clockwise
    cv::Mat_<uchar> outline_mask; //!< Outline mask of the contour.
    cv::Mat_<cv::Vec2f> U;  //!< complex vector with coordinates (x_k + i * y_k)
    cv::Mat_<cv::Vec2f> Fc; //!< DFT(U) (cartesian)
    cv::Mat_<cv::Vec2f> Fp; //!< DFT(U) (polar)

    /*!
     * \brief Determine wether a point A is less than a point B in clockwise
     *        order.
     * \param a the point A
     * \param b the point B
     * \return wether point A is less than point B in clockwise order.
     */
    bool less(cv::Point a, cv::Point b);

    void sort(); //!< Sort contour points clockwise.
};

#endif // FOURIER_DESCRIPTOR_H
