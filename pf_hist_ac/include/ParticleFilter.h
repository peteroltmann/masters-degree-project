#ifndef CONTOUR_PARTICLE_FILTER_H
#define CONTOUR_PARTICLE_FILTER_H

#include <opencv2/core/core.hpp>
#include <memory>
#include "Contour.h"
#include "Histogram.h"

class RegBasedContours;

class ParticleFilter
{
public:
    /*!
     * \brief Construct a new ContourParticleFilter with the specified number of
     * particles.
     * \param num_p the number of particles
     */
    ParticleFilter(int num_particles);
    virtual ~ParticleFilter(); //!< The default destructor.

    /*!
     * \brief Initilize particle filter.
     *
     * Call to initialize members with default values. You can also initialize
     * members bindividually.
     */
    void init(const cv::Rect templ_rect);

    /*!
     * \brief Predict state and particles.
     */
    void predict();

    /*!
     * \brief Calculate the particle weights and the mean confidence.
     */
    void calc_weight(cv::Mat &frame, cv::Size templ_size,
                     Histogram &templ_hist, float sigma);

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

    /*!
     * \brief Calculate rectangle of the state estimate or a particle.
     * \param templ_size    the size the scaling is applied to
     * \param bounds        the outer bound of the image domain
     * \param i             the particle index or -1 for state estimate
     * \return the state rectangle
     */
    cv::Rect state_rect(cv::Size templ_size, cv::Rect bounds, int i=-1);

private:
    /*!
     * \brief Calc the probability of a frame ROI by histogram comparison.
     * \param frame_roi     frame ROI whose hisogram should be compared.
     * \param templ_hist    the other histogram
     * \param sigma         deviation for probability calculation
     * \return the probability of the fame ROI matching the template.
     */
    float calc_probability(cv::Mat& frame_roi, Histogram &templ_hist,
                           float sigma);

public:
    int num_particles; //!< The number of particles.
    cv::Mat_<float> state; //!< The currently estimated state.

    std::vector<float> initial; //!< The initializing state.
    std::vector<float> sigma; //!< The standard deviations for each parameter.
    cv::Mat_<float> T; //!< The state transition matrix.

    std::vector<cv::Mat_<float>> p; //!< The particles.
    std::vector<cv::Mat_<float>> p_new; //!< The new, resampled particles.

    std::vector<float> w; //!< The weights of the particles.
    float confidence; //!< The confidence of the currently estimated state.
    std::vector<float> w_cumulative; //!< The weights for systematic resampling.
    float mean_confidence; //!< The mean confidence for systematic resampling.

    cv::RNG& rng; //!< The random number generator reference.

    float gaussian(float mu, float sigma, float x) {
        return std::exp(- pow(mu - x, 2) / pow(sigma, 2) / 2.0) /
               std::sqrt(2.0 * CV_PI * pow(sigma, 2));
    }
};

#endif // CONTOUR_PARTICLE_FILTER_H
