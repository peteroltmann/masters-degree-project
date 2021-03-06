#ifndef CONTOUR_PARTICLE_FILTER_H
#define CONTOUR_PARTICLE_FILTER_H

#include <opencv2/core/core.hpp>
#include <memory>
#include "Contour.h"

class RegBasedContours;

/*!
 * \brief Contour particle filter class.
 *
 * This class offers the functionality and data to apply a particle filter for
 * level set active contours.
 */
class ContourParticleFilter
{
public:
    /*!
     * \brief Construct a new ContourParticleFilter with the specified number of
     * particles.
     * \param num_p the number of particles
     */
    ContourParticleFilter(int num_particles);
    virtual ~ContourParticleFilter(); //!< The default destructor.

    /*!
     * \brief Initilize particle filter.
     *
     * Call to initialize members with default values. You can also initialize
     * members bindividually.
     */
    void init(const cv::Mat_<uchar> templ);

    /*!
     * \brief Predict state and particles.
     */
    void predict();

    /*!
     * \brief Calculate the particle weights and the mean confidence.
     * \param segm  the segmentation to use
     */
    void calc_weight(float templ_energy, float sigma);

    /*!
     * \brief Estimate the state.
     *
     * Estimation is done by calculating the weighted mean state.
     */
    void weighted_mean_estimate();

    /*!
     * \brief Resample particles based on their weight.
     */
    void resample();

    /*!
     * \brief Systematic resampling.
     *
     * A particle is selected repeatedly until it's confidence is less than the
     * expected cumulative confidence for that index.
     */
    void resample_systematic();

    int num_particles; //!< The number of particles.
    cv::Mat_<float> state; //!< The currently estimated state.
    Contour state_c; //!< The currently estimated state's contour parameter.

    std::vector<float> initial; //!< The initializing state.
    std::vector<float> sigma; //!< The standard deviations for each parameter.
    cv::Mat_<float> T; //!< The state transition matrix.

    std::vector<cv::Mat_<float>> p; //!< The particles.
    //!< The contour parameters of the particles.
    std::vector<std::shared_ptr<Contour>> pc;
    std::vector<cv::Mat_<float>> p_new; //!< The new, resampled particles.
    //!< The new, resampled contour parameters.
    std::vector<std::shared_ptr<Contour>> pc_new;

    std::vector<float> w; //!< The weights of the particles.
    float confidence; //!< The confidence of the currently estimated state.
    std::vector<float> w_cumulative; //!< The weights for systematic resampling.
    float mean_confidence; //!< The mean confidence for systematic resampling.

    cv::RNG& rng; //!< The random number generator reference.

    /*!
     * \brief The gaussian function.
     * \param mu    the mean of the gaussian
     * \param sigma the standard deviation of the gaussian
     * \param x     the input variable
     * \return      the value of the gaussian function at <tt>x</tt>
     */
    float gaussian(float mu, float sigma, float x) {
        return exp(- pow(mu - x, 2) / pow(sigma, 2) / 2.0) / sqrt(2.0 * CV_PI * pow(sigma, 2));
    }
};

#endif // CONTOUR_PARTICLE_FILTER_H
