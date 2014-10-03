#include "ContourParticleFilter.h"
#include "Random.h"
#include "StateParams.h"
#include "Contour.h"

#include <iostream>

ContourParticleFilter::ContourParticleFilter(int num_particles) :
    num_particles(num_particles),
    rng(Random::getRNG()),
    confidence(1.f/num_particles),
    pc_new(num_particles)
{
    // state transitions matrix with constant velocity model
    const float DT = 1;
    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0,
                                                    0, 1,  0, DT,
                                                    0, 0,  1,  0,
                                                    0, 0,  0,  1);

//    T = (cv::Mat_<float>(NUM_PARAMS, NUM_PARAMS) << 1, 0, DT,  0, 0,
//                                                    0, 1,  0, DT, 0,
//                                                    0, 0,  1,  0, 0,
//                                                    0, 0,  0,  1, 0,
//                                                    0, 0,  0,  0, 1);

    // init particle data structure
    for(int i = 0; i < num_particles; i++)
    {
       p.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       p_new.push_back(cv::Mat_<float>(NUM_PARAMS, 1));
       w.push_back(confidence); // init with uniform distributed weight
       w_cumulative.push_back(1.f);

       pc.push_back(std::shared_ptr<Contour>(new Contour));
    }
}

ContourParticleFilter::~ContourParticleFilter() {}

void ContourParticleFilter::init(const cv::Mat_<uchar> templ)
{
    initial = {0.f, 0.f, 0.f, 0.f};
    sigma = {2.f, 2.f, .5f, .5f};

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
        templ.copyTo(pc[i]->contour_mask);
    }

    // initial state
    for(int j = 0; j < NUM_PARAMS; j++)
       state.push_back(initial[j]);
}

void ContourParticleFilter::predict()
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

void ContourParticleFilter::calc_weight(float templ_energy)
{
    float sum = 0.f;
    for (int i = 0; i < num_particles; i++)
    {
        w[i] = gaussian(pc[i]->energy, 25.f, templ_energy);
//        w[i] = std::exp(- pc[i]->energy/templ_energy);
        sum += w[i];
        w_cumulative[i] = sum; // for systematic resampling
    }
    mean_confidence = sum / num_particles; // for systematic resampling

//    for (int i = 0; i < num_particles; i++)
//    {
//        w[i] /= sum;
//    }

    // TODO: CONSIDER DISTANCE FROM BEFORE TO AFTER CONTOUR EVOLUTION
}

void ContourParticleFilter::weighted_mean_estimate()
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
    pc[w_max_idx]->contour_mask.copyTo(state_c.contour_mask);
}

void ContourParticleFilter::resample()
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
//        pc_new[i] = pc[index];
    }
    p = p_new;
//    pc = pc_new;
}

void ContourParticleFilter::resample_systematic()
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
//    state_c.contour_mask.copyTo(pc_new[0]->contour_mask);
    p = p_new;
//    pc = pc_new;
}
