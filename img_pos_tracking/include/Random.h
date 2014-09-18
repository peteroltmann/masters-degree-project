#ifndef RANDOM_H
#define RANDOM_H

#include <opencv2/core/core.hpp>

class Random
{
private:
    Random();

public:
    virtual ~Random();

    static cv::RNG& getRNG();

private:
    cv::RNG rng;
};

#endif // RANDOM_H
