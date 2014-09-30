#ifndef RANDOM_H
#define RANDOM_H

#include <opencv2/core/core.hpp>

/*!
 * \brief Encapsulates opencv random number generator.
 */
class Random
{
private:
    Random(); //!< The default constructor.

public:
    virtual ~Random(); //!< The default destructor.

    /*!
     * \brief Returns the random number generator instance.
     * \return the cv::RNG instance
     */
    static cv::RNG& getRNG();

private:
    cv::RNG rng; //!< The cv::RNG instance
};

#endif // RANDOM_H
