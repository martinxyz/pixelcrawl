cmake_minimum_required(VERSION 3.8)
project(pixelcrawl)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(pybind11 2.2 REQUIRED NO_MODULE)
add_subdirectory(external/pybind11)

pybind11_add_module(pixelcrawl world/bindings.cpp)
target_compile_features(pixelcrawl PRIVATE cxx_std_14)
target_link_libraries(pixelcrawl PRIVATE Eigen3::Eigen)

target_compile_options(pixelcrawl PRIVATE -Wall -Werror -Wextra)

## Performance flags
# Measurements on my desktop:
# gcc8.2, -O2, and removing -march=native: difference ~10%
# clang7.0: 2x slower than gcc (benchmark: test_world_tick)
# debug with "-g -fno-omit-frame-pointer": ~2% slowdown

target_compile_options(pixelcrawl PRIVATE -O3 -march=native -mtune=native)
# target_compile_options(pixelcrawl PRIVATE -g -fno-omit-frame-pointer)

# only the dask framework should distribute tasks to cores
# (not sure why this is not working; still need OMP_NUM_THREADS=1 env variable)
target_compile_definitions(pixelcrawl PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(pixelcrawl PRIVATE -fno-openmp)
