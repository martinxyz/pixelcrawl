#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <iostream>
#include <cstring>
#include <Eigen/Core>

using std::vector;

constexpr int world_size = 128;
constexpr int agent_count = 200;
constexpr int agent_num_states = 6;

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

class World {
 public:
  Pixel pixels[world_size*world_size];
  vector<Agent> agents{agent_count};
  using PixelMatrix = Eigen::Matrix<uint8_t, world_size, world_size>;

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

  PixelMatrix render(int channel) {
    PixelMatrix result;
    for (int y=0; y<world_size; y++) {
      for (int x=0; x<world_size; x++) {
        result(y, x) = 0;
        auto blend = [&](Eigen::Vector3i rgb, float alpha=1.0) {
          float value = alpha * rgb[channel] + (1.0 - alpha) * result(y, x);
          result(y, x) = std::round(value);
        };
        switch (pixels[y*world_size+x].block) {
          case Food:
            blend({200, 80, 80});
            break;
          case Wall:
            blend({255, 255, 255});
            break;
        }
        if (pixels[y*world_size+x].pheromone_1) {
          blend({0, 128, 0}, 0.4);
        }
      }
    }
    for (auto &a: agents) {
      result(a.y, a.x) = 0;
      if (channel == 1) result(a.y, a.x) = 255;
    }
    return result;
  }

  void init_agents() {
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

  void tick_agents() {
    constexpr int N = world_size;
    std::uniform_int_distribution<int> d1(-1, +1);
    for (auto &a: agents) {
      pixels[a.y*N + a.x].pheromone_1 = 1;
      a.x = static_cast<unsigned>(a.x + d1(rng)) % world_size;
      a.y = static_cast<unsigned>(a.y + d1(rng)) % world_size;
      pixels[a.y*N + a.x].pheromone_1 = d1(rng);
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
