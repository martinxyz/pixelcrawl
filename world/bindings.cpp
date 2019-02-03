#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "world.hpp"
#include "render.hpp"
#include "agent.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pixelcrawl, m) {
  py::class_<World>(m, "World")
      .def(py::init<>())
      .def("seed", &World::seed)
      .def("init_map", &World::init_map)
      .def("init_agents", &World::init_agents)
      .def("tick", &World::tick)
      .def_readwrite("total_score", &World::total_score)
      ;

  m.def("render_world", &render_world);

  py::class_<AgentController>(m, "AgentController")
      .def(py::init<>())
      .def_readwrite("w0", &AgentController::w0)
      .def_readwrite("b0", &AgentController::b0)
      .def_readwrite("w1", &AgentController::w1)
      .def_readwrite("b1", &AgentController::b1)
      ;
}
