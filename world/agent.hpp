#pragma once
#include <Eigen/Core>
#include "nn.hpp"
#include <algorithm>
#include <random>

using Eigen::Matrix;

constexpr int agent_num_states = 6;
constexpr int agent_num_inputs = 11 + agent_num_states;

enum class AgentAction {Right, Down, Left, Up};

class AgentController: public SmallNN<agent_num_inputs, 20, 4+agent_num_states> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AgentAction calc(const Matrix<float, agent_num_inputs, 1> &inputs, float *states, std::default_random_engine &rng) {
    Matrix<float, 4+agent_num_states, 1> o = predict(inputs);
    using std::max;
    using std::exp;

    // First 4 outputs represent 4 actions.
    // Select a random action according to softmax probabilities.
    //
    // def softmax(x):
    //     e_x = np.exp(x - np.max(x))
    //     return e_x / e_x.sum(axis=0)
    //
    float m = max(o(0), max(o(1), max(o(2), o(3))));
    float c0 = exp(o(0)-m);
    float c1 = exp(o(1)-m);
    float c2 = exp(o(2)-m);
    float c3 = exp(o(3)-m);
    float sum = c0+c1+c2+c3;
    std::uniform_real_distribution<float> dist(0, sum);
    float roll = dist(rng);
    AgentAction action;
    if (roll < c0) action = AgentAction::Right;
    else if (roll < c0+c1) action = AgentAction::Down;
    else if (roll < c0+c1+c2) action = AgentAction::Left;
    else action = AgentAction::Up;

    constexpr float state_decay = 0.01;
    for (int i=0; i<agent_num_states; i++) {
        states[i] = (1.0-state_decay)*states[i] + state_decay*o(i+4);
    }

    return action;
  }
};
