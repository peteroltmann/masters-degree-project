#ifndef POS_PARTICLE_FILTER_H
#define POS_PARTICLE_FILTER_H

#include <opencv2/core/core.hpp>

/*!
 * \brief The StateParams enum for indexing the particle state array.
 */
enum StateParams
{
   PARAM_X,
   PARAM_Y,
   PARAM_X_VEL,
   PARAM_Y_VEL,
//   PARAM_SCALE,
   NUM_PARAMS
};

class PosParticleFilter
{
public:
    PosParticleFilter(int num_p);
    virtual ~PosParticleFilter();

    /*!
     * \brief Initilize particle filter.
     *
     * Call to initialize members with default values. You can also initialize
     * members bindividually.
     *
     * \param x_pos     the starting x coordinate
     * \param y_pos     the starting y coordinate
     */
    void init(int x_pos = 0, int y_pos = 0);

    /*!
     * \brief Predict state and particles.
     */
    void predict();

    /*!
     * \brief Calculate the particle weights and the mean confidence.
     * \param frame         the current frame
     * \param templ_size    the template rectangle size
     */
    void calcWeight(cv::Mat &frame, cv::Size templ_size);

    /*!
     * \brief Estimate the state.
     *
     * Estimation is done by calculating the weighted mean state.
     */
    void weightedMeanEstimate();

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
    void resampleSystematic();

    /*!
     * \brief Redistribute particles.
     *
     * This is supposed to be done if the target moves out of bounds, which
     * mostly happens if the target has been lost.
     */
    void redistribute(cv::Size frame_size);

    float gaussian(float mu, float sigma, float x);

    unsigned int num_p; //!< The number of particles.
    cv::Mat_<float> T; //!< The state transition matrix.
    std::vector<cv::Mat_<float>> p; //! The particles.
    std::vector<cv::Mat_<float>> p_new; //! The particles.
    std::vector<float> w; //!< The particles' weight.
    std::vector<float> w_cumulative; //! weights for systematic resampling
    float mean_confidence; //!< mean confidence for systematic resampling
    std::vector<float> sigma; //!< The sandard deviations for each parameter
    std::vector<float> initial; //!< The intializing state.
    cv::Mat_<float> state; //!< The currently estimated state.
    float confidence; //!< Confidence of the currently estimated state.
    cv::RNG& rng; //!< Random number generator reference;

};

#endif // POS_PARTICLE_FILTER_H
