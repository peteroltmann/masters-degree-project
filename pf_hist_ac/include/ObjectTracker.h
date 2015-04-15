#ifndef OBJECT_TRACKER_H
#define OBJECT_TRACKER_H

#include <opencv2/core/core.hpp>
#include <string>

/*!
 * \brief The Object Tracker class. Run the 'Object Tracking with Particle
 *        Filters and Actice Contours' algorithm.
 */
class ObjectTracker
{
public:
    ObjectTracker(); //!< The default constructor
    virtual ~ObjectTracker(); //!< The default destructor

    /*!
     * \brief Run the 'Object Tracking with Particle Filters and Actice
     *        Contours' algorithm.
     * \param param_path    path to parameterization file
     * \return main exit code
     */
    static int run(std::string param_path);

private:
    /*!
     * \brief Find zero-padding free rectangle/roi for the given image
     * \param   img a zero paddded image
     * \return the rectangle describing the zero-padding free roi of the image
     */
    static cv::Rect padding_free_rect(cv::Mat img);
};

#endif // OBJECTTRACKER_H
