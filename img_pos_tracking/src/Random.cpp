#include "Random.h"
#include <ctime>

Random::Random() :
    rng(std::time(NULL))
{}

Random::~Random() {}

cv::RNG& Random::getRNG()
{
    static Random instance;
    return instance.rng;
}
