#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "world.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pixelcrawl, m) {
  py::class_<World>(m, "World")
      .def(py::init<>())
      .def("seed", &World::seed)
      .def("init_map", &World::init_map)
      .def("init_agents", &World::init_agents)
      .def("render", &World::render)
      .def("tick", &World::tick);

      // .def_readwrite("pixels", &World::pixels);

  py::class_<Agent>(m, "Agent")
      .def(py::init<>())
      .def_readwrite("x", &Agent::x)
      .def_readwrite("y", &Agent::y);

  /*
  // bad idea: address of bit fields (also, depends on memory layout/implementation, which should be private)
  // instead, do something like setWalls(array), clearWalls() etc. or get/setMap(Food, array)
  py::class_<Pixel>(m, "Pixel")
      .def_readwrite("block", &Pixel::block)
      .def_readwrite("pheromone_1", &Pixel::pheromone_1)
      .def_readwrite("pheromone_2", &Pixel::pheromone_2)
      .def_readwrite("reserved_1", &Pixel::reserved_1)
      .def_readwrite("reserved_2", &Pixel::reserved_2)
  };
  */
}
