#include "ParticleFilter.h"
#include "Random.h"
#include "StateParams.h"
#include "Contour.h"
#include "hist.h"

#include <iostream>

ParticleFilter::ParticleFilter(int num_particles) :
    num_particles(num_particles),
    rng(Random::getRNG()),
    confidence(1.f/num_particles)
{
    // state transitions matrix with constant velocity model
    const float DT = 1;
//    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0,
//                                                    0, 1,  0, DT,
//                                                    0, 0,  1,  0,
//                                                    0, 0,  0,  1);

    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0, 0,
                                                    0, 1,  0, DT, 0,
                                                    0, 0,  1,  0, 0,
                                                    0, 0,  0,  1, 0,
                                                    0, 0,  0,  0, 1);

    // init particle data structure
    for(int i = 0; i < num_particles; i++)
    {
       p.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       p_new.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       w.push_back(confidence); // init with uniform distributed weight
       w_cumulative.push_back(1.f);
    }
}

ParticleFilter::~ParticleFilter() {}

void ParticleFilter::init(const cv::Rect templ_rect)
{
    // [x, y, x_vel, y_vel, scale]
    initial = {templ_rect.x + templ_rect.width/2.f,
               templ_rect.y + templ_rect.height/2.f, 0.f, 0.f, 1.f};
    sigma = {2.f, 2.f, .5f, .5f, .1f}; // distortion deviation

    std::cout << "Init with state: [ ";
    for( int j = 0; j < NUM_PARAMS; j++)
       std::cout << initial[j] << " ";
    std::cout << "]" << std::endl;

    // init particles
    for (int i = 0; i < num_particles; i++)
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

void ParticleFilter::predict()
{
    state = T * state; // predict new state

    for (int i = 0; i < num_particles; i++)
    {
        cv::Mat_<float> noise(NUM_PARAMS, 1);
        for (int j = 0; j < NUM_PARAMS; j++)
            noise(j) = rng.gaussian(sigma[j]); // calc noise vector

        p[i] = T * p[i] + noise; // predict particles
    }
}

void ParticleFilter::calc_weight(cv::Mat& frame, cv::Size templ_size,
                                 cv::Mat_<float>& templ_hist, float sigma)
{
    cv::Rect bounds(0, 0, frame.cols, frame.rows);

    // estimated state confidence
    float scale = std::max(.1f, state(PARAM_SCALE));
    state(PARAM_SCALE) = scale;
    int width = std::round(templ_size.width * scale);
    int height = std::round(templ_size.height * scale);
    int x = std::round(state(PARAM_X)) - width / 2;
    int y = std::round(state(PARAM_Y)) - height / 2;

    cv::Rect region = cv::Rect(x, y, width, height) & bounds;
    cv::Mat frame_roi(frame, region);

    confidence = calc_probability(frame_roi, templ_hist, sigma);

    // particle confidence
    float sum = 0.f;
    for (int i = 0; i < num_particles; i++)
    {
        float scale = std::max(.1f, p[i](PARAM_SCALE));
        p[i](PARAM_SCALE) = scale;
        int width = std::round(templ_size.width * scale);
        int height = std::round(templ_size.height * scale);
        int x = std::round(p[i](PARAM_X)) - width / 2;
        int y = std::round(p[i](PARAM_Y)) - height / 2;

        cv::Rect region = cv::Rect(x, y, width, height) & bounds;
        cv::Mat frame_roi(frame, region);

        w[i] = calc_probability(frame_roi, templ_hist, sigma);
        sum += w[i];
        w_cumulative[i] = sum; // for systematic resampling

    }
    mean_confidence = sum / num_particles; // for systematic resampling
}

void ParticleFilter::weighted_mean_estimate()
{
    float sum = 0.f;
    float w_max = 0.f;
    int w_max_idx = 0.f;
    cv::Mat_<float> tmp = cv::Mat_<float>::zeros(NUM_PARAMS, 1);
    for (int i = 0; i < num_particles; i++)
    {
        tmp += p[i] * w[i];
        sum += w[i];
        if (w_max < w[i])
        {
            w_max = w[i];
            w_max_idx = i;
        }
    }
    state = tmp / sum;
}

void ParticleFilter::resample()
{
    int index = Random::getRNG().uniform(0, num_particles);
    float beta = 0.f;
    float w_max = 0.f;
    for (int i = 0; i < num_particles; i++)
    {
        if (w[i] > w_max)
            w_max = w[i];
    }

    for (int i = 0; i < num_particles; i++)
    {
        beta += Random::getRNG().uniform(0.f, 2.f * w_max);
        while (beta > w[index])
        {
            beta -= w[index];
            index = (index + 1) % num_particles;
        }
        p[index].copyTo(p_new[i]);
    }
    p = p_new;
}

void ParticleFilter::resample_systematic()
{
    for(int i = 0; i < num_particles; i++)
    {
        int j = 0;
        while((w_cumulative[j] <= (float) i * mean_confidence) &&
              (j < num_particles-1))
        {
            j++;
        }
        p[j].copyTo(p_new[i]);
//        pc_new[i] = pc[j];
    }
    // Since particle 0 always gets chosen by the above,
    // assign the mean state to it
    state.copyTo(p_new[0]);
    p = p_new;
}

float ParticleFilter::calc_probability(cv::Mat &frame_roi,
                                       cv::Mat_<float> &templ_hist, float sigma)
{
    static cv::Mat_<float> hist;

    calc_hist(frame_roi, hist);
    normalize(hist);
    float bc = calcBC(templ_hist, hist);

    float prob = 0.f;
    if (bc <= 1.f) // total missmatch
        prob = std::exp(-sigma * bc*bc);
    return prob;
}
