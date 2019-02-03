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

cmake .
make -s

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
