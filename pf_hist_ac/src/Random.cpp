#include "Random.h"
#include <ctime>

Random::Random() :
    rng(std::time(NULL)) // use current time as seed for random generator
{}

Random::~Random() {}

cv::RNG& Random::getRNG()
{
    static Random instance;
    return instance.rng;
}
