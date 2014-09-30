#include "PosParticleFilter.h"
#include "Random.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

PosParticleFilter::PosParticleFilter(int num_p) :
    num_p(num_p),
    rng(Random::getRNG()),
    confidence(1.f/num_p)
{
    // state transitions matrix with constant velocity model
    const float DT = 1;
    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0,
                                                    0, 1,  0, DT,
                                                    0, 0,  1,  0,
                                                    0, 0,  0,  1);
//    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0,  0,
//                                                    0, 1,  0, DT,  0,
//                                                    0, 0,  1,  0,  0,
//                                                    0, 0,  0,  1,  0,
//                                                    0, 0,  0,  0,  1);
    // init particle data structure
    for(int i = 0; i < num_p; i++)
    {
       p.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       p_new.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       w.push_back(confidence); // init with uniform distributed weight
       w_cumulative.push_back(1.f);
    }
}

PosParticleFilter::~PosParticleFilter() {}

void PosParticleFilter::init(int x_pos, int y_pos)
{

    // set default values
    initial = {(float) x_pos, (float) y_pos, 0.f, 0.f/*, 1.f*/};
    sigma = {2.f, 2.f, .5f, .5f/*, .1f*/};

    std::cout << "Init with state: [ ";
    for( int j = 0; j < NUM_PARAMS; j++)
       std::cout << initial[j] << " ";
    std::cout << "]" << std::endl;

    // init particles
    for (int i = 0; i < num_p; i++)
    {
        for (int j = 0; j < NUM_PARAMS; j++)
        {
            float noise = rng.gaussian(sigma[j]);
            p[i](j) = initial[j] + noise;
        }
    }

    // initial state
    for(int j = 0; j < NUM_PARAMS; j++)
       state.push_back(initial[j]);

}

void PosParticleFilter::predict()
{
    state = T * state; // predict new state

    for (int i = 0; i < num_p; i++)
    {
        cv::Mat_<float> noise(NUM_PARAMS, 1);
        for (int j = 0; j < NUM_PARAMS; j++)
            noise(j) = rng.gaussian(sigma[j]); // calc noise vector

        p[i] = T * p[i] + noise; // predict particles
    }
}

void PosParticleFilter::calcWeight(cv::Mat& frame, cv::Size templ_size)
{
    cv::Rect bounds(0, 0, frame.cols, frame.rows);
    float sense_noise = 2.f;
    float max_white_pixel = templ_size.width * templ_size.height; // perfect fit

    // estimated state confidence
    int width = templ_size.width;
    int height = templ_size.height;
    int x = std::round(state(PARAM_X)) - width / 2;
    int y = std::round(state(PARAM_Y)) - height / 2;

    cv::Rect region = cv::Rect(x, y, width, height) & bounds;
    cv::Mat frame_roi(frame, region);

    confidence = cv::countNonZero(frame_roi) / max_white_pixel;

    // particle confidence
    float sum = 0.f;
    for (int i = 0; i < num_p; i++)
    {
        int width = templ_size.width;
        int height = templ_size.height;
        int x = std::round(p[i](PARAM_X)) - width / 2;
        int y = std::round(p[i](PARAM_Y)) - height / 2;

        cv::Rect region = cv::Rect(x, y, width, height) & bounds;
        cv::Mat frame_roi(frame, region);

        w[i] = cv::countNonZero(frame_roi) / max_white_pixel;
//        w[i] = gaussian(1.f, .4f, cv::countNonZero(frame_roi) / max_white_pixel);
        sum += w[i];
        w_cumulative[i] = sum; // for systematic resampling

    }
    mean_confidence = sum / num_p; // for systematic resampling

    // Zur Berechnung der Likelyhood wird hier kein Vergleich zu einer Messung
    // verwendet, sondern ein Vergleich der Histogramme mit einem Template
    // (oder eben der Energien und Abstände)

    // Zur Simulation deshalb die Idee: weißes Rechteck suchen in schwarzem
    // Bild. Dafür sollte dann doch die Zustandsparameter um die Skalierung
    // erweitert werden.
}

void PosParticleFilter::weightedMeanEstimate()
{
    float sum = 0.f;
    cv::Mat_<float> tmp = cv::Mat_<float>::zeros(NUM_PARAMS, 1);
    for (int i = 0; i < num_p; i++)
    {
        tmp += p[i] * w[i];
        sum += w[i];

    }
    state = tmp / sum;

}

void PosParticleFilter::resample()
{

    int index = Random::getRNG().uniform(0, num_p);
    float beta = 0.f;
    float w_max = 0.f;
    for (int i = 0; i < num_p; i++)
    {
        if (w[i] > w_max)
            w_max = w[i];
    }

    for (int i = 0; i < num_p; i++)
    {
        beta += Random::getRNG().uniform(0.f, 2.f * w_max);
        while (beta > w[index])
        {
            beta -= w[index];
            index = (index + 1) % num_p;
        }
        p[index].copyTo(p_new[i]);
    }
    p = p_new;
}

void PosParticleFilter::resampleSystematic()
{
    for(int i = 0; i < num_p; i++)
    {
        int j = 0;
        while((w_cumulative[j] <= (float) i * mean_confidence) && (j < num_p-1))
        {
            j++;
        }
        p[j].copyTo(p_new[i]);
    }
    // Since particle 0 always gets chosen by the above,
    // assign the mean state to it
    state.copyTo(p_new[0]);
    p = p_new;
}

void PosParticleFilter::redistribute(cv::Size frame_size)
{
    static const float lower_bound[NUM_PARAMS] = {0, 0, -.5, -.5};
    static const float upper_bound[NUM_PARAMS] = {(float) frame_size.width,
                                                  (float) frame_size.height,
                                                  .5, .5};

    std::cout << "Redistribute: " << state << " - " << confidence << std::endl;

    for (int i = 0; i < num_p; i++)
    {
        for (int j = 0; j < NUM_PARAMS; j++)
        {
            p[i](j) = Random::getRNG().uniform(lower_bound[j], upper_bound[j]);
        }
    }
}

float PosParticleFilter::gaussian(float mu, float sigma, float x)
{
    return std::exp(- std::pow(mu - x, 2) / std::pow(sigma, 2) / 2.0)
            / std::sqrt(2.0 * CV_PI * std::pow(sigma, 2));
}
