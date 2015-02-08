#ifndef CONTOUR_EVOLUTION_H
#define CONTOUR_EVOLUTION_H

#include <string>

/*!
 * \brief Contour Evolution class. Run a single image contour evolution.
 */
class ContourEvolution
{
public:
    ContourEvolution(); //!< The default constructor
    virtual ~ContourEvolution(); //!< The default destructor

    /*!
     * \brief Run a single image contour evolution
     * \param param_path    path to parameterization file
     * \return main exit code
     */
    static int run(std::string param_path);
};

#endif // CONTOUREVOLUTION_H
