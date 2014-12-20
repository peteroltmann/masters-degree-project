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
    }
}

ParticleFilter::~ParticleFilter() {}

void ParticleFilter::init(const cv::Mat_<uchar> templ)
{
    // calculate mass center of template contour (translation from [0, 0])
    cv::Moments m = cv::moments(templ, true);
    cv::Point2f center(m.m10/m.m00, m.m01/m.m00);

    initial = {center.x, center.y, 0.f, 0.f};
    sigma = {2.f, 2.f, .5f, .5f}; // distortion deviation

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

void ParticleFilter::calc_weight(cv::Mat& frame, cv::Mat_<uchar> templ,
                                 cv::Mat_<float>& templ_hist, float sigma)
{
    cv::Rect bounds(0, 0, frame.cols, frame.rows);

    // estimated state confidence
    // transform template contour (translation only) and calc hist

    cv::Moments m = cv::moments(templ, true);
    cv::Point2f center(m.m10/m.m00, m.m01/m.m00);
    cv::Mat_<float> templ_at_zero = (cv::Mat_<float>(4, 1) <<
                                  -center.x, -center.y, 0, 0);
    cv::Mat_<float> tmp = templ_at_zero + state;

    templ.copyTo(state_c.contour_mask);
    state_c.transform_affine(tmp);

//    cv::Rect region = cv::Rect(x, y, width, height) & bounds;
//    cv::Mat frame_roi(frame, region);

    // TODO
    confidence = calc_probability(frame, templ_hist, state_c.contour_mask,
                                  sigma);

    // particle confidence
    float sum = 0.f;
    for (int i = 0; i < num_particles; i++)
    {
        cv::Mat_<float> tmp = templ_at_zero + p[i];

        Contour pi_c;
        templ.copyTo(pi_c.contour_mask);
        pi_c.transform_affine(tmp);

        // TODO calc histogram only in a ROI around the contour for faster
        // computation
//        cv::Rect region = cv::Rect(x, y, width, height) & bounds;
//        cv::Mat frame_roi(frame, region);

        w[i] = calc_probability(frame, templ_hist, pi_c.contour_mask, sigma);
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
                                       cv::Mat_<float> &templ_hist,
                                       cv::Mat& mask, float sigma)
{
    static cv::Mat_<float> hist;

    calc_hist(frame_roi, hist, mask);
//    cv::normalize(hist, hist);
//    float bc = cv::compareHist(templ_hist, hist, CV_COMP_BHATTACHARYYA);
    normalize(hist);
    float bc = calcBC(templ_hist, hist);

    float prob = 0.f;
    if (bc <= 1.f) // total missmatch
        prob = std::exp(-sigma * bc*bc);
//        prob = 1 - bc;
    return prob;
}
