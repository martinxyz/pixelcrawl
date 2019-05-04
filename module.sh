# sourced by other script
# compile python module, prepare environment
set -e

# if ! [ -d build ]; then
#     mkdir build
#     cmake -B build .
# fi
# make -C build VERBOSE=1
# make -s -C build
# export PYTHONPATH="build"

# rm -rf CMakeFiles CMakeCache.txt
cmake .

# make -s
make VERBOSE=1

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
