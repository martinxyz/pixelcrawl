# pixelcrawl
A sandbox for artificial life and evolution strategies.

See article: [Exploring a Pixel-Maze with Evolution Strategies](https://log2.ch/2019/exploring-a-pixel-maze-with-evolution-strategies)

The code consists of a Python part and a Python module written in C++.

## Compile and Run

Python dependencies:
```
pip install -r requirements.txt  # preferably in a virtualenv
```

C++ dependencies:
```
sudo apt install libeigen3-dev  # or equivalent
git submodule update  # fetches external/pybind11
```

The `./train.sh` script will invoke cmake (creating build-rel and build-dbg), compile the module, and then call `train.py`. If you want to call python scripts directly instead, you need to set PYTHONPATH to include the build directory (see `module.sh`).

```
./train.sh --help
./train.sh print_config
./train.sh -o test-run
./render.sh test-run world_count=2 world_ticks=400 render=test-run/xbest.dat
```

(If you get a strange Python error about paths you probably have the wrong sacred version, see above.)

This trains a quick variant with default parameters (200 agents) which takes about 40 minutes on a low-end PC. In case someone is really doing this, while you are waiting, please keep in mind that the original Python version took about 100 times longer (3 days) for exactly the same result. The render step can be started while the training is still running. The commandline interface comes mostly from [sacred](https://github.com/IDSIA/sacred).

## Reproducing Article Results

To get results similar to those in the article, but much faster:

```
./train.sh -o outputs/easy-hunting with easy-hunting-config.json
```

This does 50'000 evaluations, which takes about one hour on a low-end PC and reaches a mean reward of approximately 1000.

The actual command line for the videos in the article:

```
./train.sh -o outputs/long-hunting with easy-hunting-config.json cmaes_popsize=200 evaluations=2000000 mapgen.agent_n_hidden=40 world_count=100
```
    
which is complete overkill and will take days. Especially the world_count: it costs much and doesn't really help. (I was just throwing CPUs at it to see what happens.) The run was based on revision a42c78b8c. Most likely similar results can still be reproduced with the current version.

Note: I've hardcoded the size of the hidden layer because Eigen is substantially faster when this is known at compile-time. You'll have to change it in the code when you run into the assertion.
