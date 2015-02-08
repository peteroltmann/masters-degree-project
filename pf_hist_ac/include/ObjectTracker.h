#ifndef OBJECT_TRACKER_H
#define OBJECT_TRACKER_H

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
};

#endif // OBJECTTRACKER_H
