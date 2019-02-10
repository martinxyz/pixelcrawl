#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <chrono>
#include <iostream>
#include <cstring>
#include <Eigen/Core>
#include "agent.hpp"

using std::vector;

constexpr int world_size = 128;

enum Block {
  Empty = 0,
  Wall,
  Food,
};

struct Pixel {
  unsigned int block : 4;
  unsigned int pheromone_1 : 1;
  unsigned int pheromone_2 : 1;
  unsigned int reserved_1 : 1;
  unsigned int reserved_2 : 1;
};

struct Agent {
  int x, y;
  float state[agent_num_states];
};

using PixelMatrix = Eigen::Matrix<uint8_t, world_size, world_size>;

class World {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int total_score_ = 0;
  Pixel pixels_[world_size*world_size];
  vector<Agent> agents_;

  explicit World(unsigned int seed) {
    rng_.seed(seed);
  }

  void init_map(PixelMatrix walls, PixelMatrix food) {
    std::memset(pixels_, 0, sizeof(pixels_));
    for (int y=0; y<world_size; y++) {
      for (int x=0; x<world_size; x++) {
        Block b = Empty;
        if (food(y, x) != 0) b = Food;
        if (walls(y, x) != 0) b = Wall;
        pixels_[y*world_size+x].block = b;
      }
    }
  }

  void init_agents(const AgentController &ac, int agent_count, bool easy_start, float walk_through_wall_prob) {
    walk_through_wall_prob_ = walk_through_wall_prob;
    agentController_ = ac;
    constexpr int N = world_size;
    int spread = easy_start ? N/4 : N/12;
    std::uniform_int_distribution<int> pos_dist(N/2 - spread, N/2 + spread);
    std::normal_distribution<float> state_dist(0.0, 1.0);

    std::default_random_engine rng_fixed;
    rng_fixed.seed(1);
    std::default_random_engine &rng_init = easy_start ? rng_fixed : rng_;

    for (int i=0; i<agent_count; i++) {
      Agent a;
      do {
        a.x = pos_dist(rng_init);
        a.y = pos_dist(rng_init);
      } while (easy_start && pixel_at(a.x, a.y).block != Empty);
      for (auto &v : a.state) {
        v = state_dist(rng_init);
      }
      agents_.push_back(a);
    }
  }

  void tick() {
    tick_agents();
    tick_pheromones();
  }

 private:
  AgentController agentController_;
  std::default_random_engine rng_;
  float walk_through_wall_prob_;

  inline Pixel& pixel_at(int x, int y) {
    unsigned int x_ = static_cast<unsigned>(x) % world_size;
    unsigned int y_ = static_cast<unsigned>(y) % world_size;
    return pixels_[y_*world_size+x_];
  }

  void tick_agents() {
    constexpr int N = world_size;
    std::normal_distribution<float> normal_dist(0.0, 1.0);
    for (auto &a: agents_) {
      pixels_[a.y*N + a.x].pheromone_1 = 1;

      Eigen::Matrix<float, agent_num_inputs, 1> inputs;
      int idx = 0;
      // see Wall
      inputs(idx++) = pixel_at(a.x+0, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+1, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x-1, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y+1).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y-1).block == Wall ? 1 : 0;
      // see Food
      // inputs(idx++) = pixel_at(a.x+0, a.y+0).block == Food ? 1 : 0;  // always 0
      inputs(idx++) = pixel_at(a.x+1, a.y+0).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x-1, a.y+0).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y+1).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y-1).block == Food ? 1 : 0;
      // smell Pheromone
      int count_pheremone_1 = 0;
      // count_pheremone_1 += pixel_at(a.x+0, a.y+0).pheromone_1;  // always 1
      count_pheremone_1 += pixel_at(a.x+1, a.y+0).pheromone_1;
      count_pheremone_1 += pixel_at(a.x-1, a.y+0).pheromone_1;
      count_pheremone_1 += pixel_at(a.x+0, a.y+1).pheromone_1;
      count_pheremone_1 += pixel_at(a.x+0, a.y-1).pheromone_1;
      inputs(idx++) = count_pheremone_1 / 5.0;
      // know previous state
      for (int i=0; i<agent_num_states; i++) inputs[idx++] = a.state[i];
      // one random input
      inputs(idx++) = normal_dist(rng_);

      // assert that cannot be turned off (but probably will be optimized away)
      if (idx != agent_num_inputs) {
        std::cout << "produced " << idx << " inputs, expected " << agent_num_inputs << std::endl;
        exit(1);
      }

      AgentAction action = agentController_.calc(inputs, a.state, rng_);

      int dx = 0;
      int dy = 0;
      switch (action) {
        case AgentAction::Right: dx = +1; break;
        case AgentAction::Down:  dy = +1; break;
        case AgentAction::Left:  dx = -1; break;
        case AgentAction::Up:    dy = -1; break;
      }

      std::bernoulli_distribution walk_through_wall(walk_through_wall_prob_);
      if (pixel_at(a.x + dx, a.y + dy).block != Wall || walk_through_wall(rng_)) {
        a.x = static_cast<unsigned>(a.x + dx) % world_size;
        a.y = static_cast<unsigned>(a.y + dy) % world_size;
      }

      pixels_[a.y*N + a.x].pheromone_1 = 1;
      if (pixels_[a.y*N + a.x].block == Food) {
        pixels_[a.y*N + a.x].block = Empty;
        total_score_++;
      }
    }
  }

  void tick_pheromones() {
    std::bernoulli_distribution dist(1.0/128);
    for (auto &p: pixels_) {
      if (p.pheromone_1 && dist(rng_)) {
        p.pheromone_1 = 0;
      }
    }
    // std::random_shuffle(agents_.begin(), agents_.end());
  }
};
