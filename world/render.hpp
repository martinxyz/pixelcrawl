#pragma once
#include "world.hpp"
#include <Eigen/Core>
#include <cstdint>

using RenderBuffer = Eigen::Matrix<uint8_t, world_size, world_size>;

RenderBuffer render_world(const World &world, int channel) {
  RenderBuffer result;
  for (int y=0; y<world_size; y++) {
    for (int x=0; x<world_size; x++) {
      result(y, x) = 0;
      auto blend = [&](Eigen::Vector3i rgb, float alpha=1.0) {
        float value = alpha * rgb[channel] + (1.0 - alpha) * result(y, x);
        result(y, x) = std::round(value);
      };
      switch (world.pixels[y*world_size+x].block) {
        case Food: blend({200, 80, 80}); break;
        case Wall: blend({255, 255, 255}); break;
      }
      if (world.pixels[y*world_size+x].pheromone_1) {
        blend({0, 128, 0}, 0.4);
      }
    }
  }
  for (auto &a: world.agents) {
    result(a.y, a.x) = 0;
    if (channel == 1) result(a.y, a.x) = 255;
  }
  return result;
}
