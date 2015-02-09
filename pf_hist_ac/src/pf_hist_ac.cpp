#include "ObjectTracker.h"
#include "ContourEvolution.h"

#include <iostream>
#include <string>
#include <algorithm>

char* get_cmd_option(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) // found and next arg exists
        return *itr; // return next arg

    return 0; // not found or no next arg
}

bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char* argv[])
{
    // =========================================================================
    // = PARSE ARGUMENTS                                                         =
    // =========================================================================

    std::string param_path("../parameterization.yml");
    bool ce_only  = false; // true: use only single image cotnour evolution

    char** begin = argv;
    char** end = argv+argc;

    if (cmd_option_exists(begin, end, "-h"))
    {
        std::cout << "Description:" << std::endl;
        std::cout << "  Tracking Objects with Particle Filters and Active "
                     "Contours" << std::endl << std::endl;

        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " [-f <param-file>] [-c]"
                  << std::endl << std::endl;

        std::cout << "Options:" << std::endl;
        std::cout << "  -f <param-file>  Specify parameterization file, "
                     "default: ../parameterization.yml" << std::endl;
        std::cout << "  -c               Use only single image contour "
                     "evoulution" << std::endl << std::endl;

        std::cout << "Control:" << std::endl;
        std::cout << "  Press 'q' to quit" << std::endl;
        std::cout << "  Press 'space' pause/resume" << std::endl;

        return EXIT_SUCCESS;
    }

    ce_only = cmd_option_exists(begin, end, "-c");

    char* arg_f = get_cmd_option(begin, end, "-f");
    if (arg_f)
        param_path = arg_f;


    // =========================================================================
    // = RUN APPLICATION
    // =========================================================================

    if (ce_only)
        return ContourEvolution::run(param_path);
    else
        return ObjectTracker::run(param_path);

}
