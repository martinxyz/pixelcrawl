#!/bin/bash
source module.sh
make -C "$BUILD_DIR" test CTEST_OUTPUT_ON_FAILURE=1
python3 -m pytest world/test_world.py::test_world_tick \
        --benchmark-columns='min, median, max, rounds, iterations'
