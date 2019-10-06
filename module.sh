# sourced by other script
# compile python module, prepare environment
set -e

# clean up old in-source build (if still present)
rm -rf CMakeFiles CMakeCache.txt Makefile *.so

# rm -rf build build-dbg build-rel
if [[ ! -d build-dbg ]]; then
    cmake -DCMAKE_BUILD_TYPE=Debug -S . -B build-dbg
fi
if [[ ! -d build-rel ]]; then
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build-rel
fi

# cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Debug -S . -B build-clang

BUILD_DIR=build-dbg

# make -C "$BUILD_DIR" clean
# make -s -C "$BUILD_DIR"
make -C "$BUILD_DIR" VERBOSE=1
# make -C "$BUILD_DIR" test CTEST_OUTPUT_ON_FAILURE=1
export PYTHONPATH="$(pwd)/$BUILD_DIR"

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
