#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <iostream>
#include <cstring>
#include <Eigen/Core>
#include "agent.hpp"

using std::vector;

constexpr int world_size = 128;
constexpr int agent_count = 200;

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
  Pixel pixels[world_size*world_size];
  vector<Agent> agents{agent_count};
  AgentController agentController;
  int total_score = 0;

  void seed(unsigned int seed) {
    rng.seed(seed);
  }

  void init_map(PixelMatrix walls, PixelMatrix food) {
    std::memset(pixels, 0, sizeof(pixels));
    for (int y=0; y<world_size; y++) {
      for (int x=0; x<world_size; x++) {
        Block b = Empty;
        if (food(y, x) != 0) b = Food;
        if (walls(y, x) != 0) b = Wall;
        pixels[y*world_size+x].block = b;
      }
    }
  }

  void init_agents(const AgentController &ac) {
    agentController = ac;
    constexpr int N = world_size;
    std::uniform_int_distribution<int> pos_dist(N/2 - N/12, N/2 + N/12);
    std::normal_distribution<float> state_dist(0.0, 1.0);
    for (auto &a : agents) {
      a.x = pos_dist(rng);
      a.y = pos_dist(rng);
      for (auto &v : a.state) {
        v = state_dist(rng);
      }
    }
  }

  void tick() {
    tick_agents();
    tick_pheromones();
  }

 private:
  std::default_random_engine rng;

  inline Pixel& pixel_at(int x, int y) {
    unsigned int x_ = static_cast<unsigned>(x) % world_size;
    unsigned int y_ = static_cast<unsigned>(y) % world_size;
    return pixels[y_*world_size+x_];
  }

  void tick_agents() {
    constexpr int N = world_size;
    std::normal_distribution<float> normal_dist(0.0, 1.0);
    for (auto &a: agents) {
      pixels[a.y*N + a.x].pheromone_1 = 1;

      Eigen::Matrix<float, agent_num_inputs, 1> inputs;
      int idx = 0;
      // see Wall
      inputs(idx++) = pixel_at(a.x+0, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+1, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x-1, a.y+0).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y+1).block == Wall ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y-1).block == Wall ? 1 : 0;
      // see Food
      inputs(idx++) = pixel_at(a.x+0, a.y+0).block == Food ? 1 : 0;  // actually, always 0
      inputs(idx++) = pixel_at(a.x+1, a.y+0).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x-1, a.y+0).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y+1).block == Food ? 1 : 0;
      inputs(idx++) = pixel_at(a.x+0, a.y-1).block == Food ? 1 : 0;
      // smell Pheromone
      int count_pheremone_1 = 0;
      count_pheremone_1 += pixel_at(a.x+0, a.y+0).pheromone_1;  //probably always 1?
      count_pheremone_1 += pixel_at(a.x+1, a.y+0).pheromone_1;
      count_pheremone_1 += pixel_at(a.x-1, a.y+0).pheromone_1;
      count_pheremone_1 += pixel_at(a.x+0, a.y+1).pheromone_1;
      count_pheremone_1 += pixel_at(a.x+0, a.y-1).pheromone_1;
      inputs(idx++) = count_pheremone_1 / 5.0;
      // know previous state
      for (int i=0; i<agent_num_states; i++) inputs[idx++] = a.state[i];
      // one random input
      inputs(idx++) = normal_dist(rng);

      // assert that cannot be turned off (but probably will be optimized away)
      if (idx != agent_num_inputs) {
        std::cout << "idx:" << idx << std::endl;
        abort();
      }

      AgentAction action = agentController.calc(inputs, rng);

      int dx = 0;
      int dy = 0;
      switch (action) {
        case AgentAction::Right: dx = +1; break;
        case AgentAction::Down:  dy = +1; break;
        case AgentAction::Left:  dx = -1; break;
        case AgentAction::Up:    dy = -1; break;
      }

      // std::uniform_int_distribution<int> d1(-1, +1);
      // int dx = d1(rng);
      // int dy = d1(rng);

      std::bernoulli_distribution walk_into_wall(1.0/16);
      if (pixel_at(a.x + dx, a.y + dy).block != Wall || walk_into_wall(rng)) {
        a.x = static_cast<unsigned>(a.x + dx) % world_size;
        a.y = static_cast<unsigned>(a.y + dy) % world_size;
      }
      pixels[a.y*N + a.x].pheromone_1 = 1;
      if (pixels[a.y*N + a.x].block == Food) {
        pixels[a.y*N + a.x].block = Empty;
        total_score++;
      }
    }
  }

  void tick_pheromones() {
    std::bernoulli_distribution dist(1.0/128);
    for (auto &p: pixels) {
      if (p.pheromone_1 && dist(rng)) {
        p.pheromone_1 = 0;
      }
    }
    // std::random_shuffle(agents.begin(), agents.end());
  }
};
