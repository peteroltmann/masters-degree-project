#ifndef FOURIER_DESCRIPTOR_H
#define FOURIER_DESCRIPTOR_H

#include <opencv2/core/core.hpp>
#include <complex>

/*!
 * \brief Fourier descriptor 'complex coordinates'.
 */
class FourierDescriptor
{
public:

    FourierDescriptor(); //!< The default constructor.

    /*!
     * \brief Construct fourier descriptor from a given contour mask.
     * \param mask  masks interior pixel (interior = 1/ != 0)
     */
    FourierDescriptor(const cv::Mat_<uchar>& mask);

    FourierDescriptor(const FourierDescriptor& other); //!< Copy constructor.

    //! Assignement operator.
    FourierDescriptor& operator=(const FourierDescriptor& other);

    virtual ~FourierDescriptor(); //!< The default destructor.

    /*!
     * \brief init  Initialize fourier descriptor from a given contour mask.
     * \param mask  masks interior pixel (interior = 1/ != 0)
     */
    void init(const cv::Mat_<uchar>& mask);

    /*!
     * \brief Match with another fourier descriptor.
     *
     * Uses normalization to obtain invariance against translation, scale and
     * rotation.
     *
     * \param fd2   the other fourier descriptor
     * \return
     */
    float match(const FourierDescriptor& fd2);

    /*!
     * \brief Reconstruct the contour shape from the descriptor.
     * \return the reconstructed contour shape.
     */
    cv::Mat_<uchar> reconstruct();

    /*!
     * \brief Filter low frequencies.
     *
     * This is done by setting the high frequencies to zero.
     *
     * \param num_fourier   keep the <tt>2*num_fourier</tt> lowest frequencies.
     */
    void low_pass(int num_fourier);

private:
    cv::Size mask_size; //!< the size of the contour mask for reconstruction
    int num_points; //!< number of contour points
    std::vector<cv::Point> cp; //!< contour points sorted clockwise
    cv::Mat_<cv::Vec2f> U;  //!< complex vector with coordinates (x_k + i * y_k)
    cv::Mat_<cv::Vec2f> Fc; //!< DFT(U) (cartesian)
    cv::Mat_<cv::Vec2f> Fp; //!< DFT(U) (polar)
};

#endif // FOURIER_DESCRIPTOR_H
