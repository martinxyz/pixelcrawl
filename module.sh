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

rm -rf CMakeFiles CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Release .
# cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .
# cmake -DCMAKE_BUILD_TYPE=Debug .
# cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Debug .

make clean
make VERBOSE=1

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
