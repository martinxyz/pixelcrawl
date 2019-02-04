import numpy as np
import mapgen
from sacred import Experiment

def test_world_tick(benchmark):
    ex = Experiment('pytest', ingredients=[mapgen.ing])
    @ex.command
    def f():
        param_count = mapgen.count_params()
        np.random.seed(1)
        params = np.random.randn(param_count)
        world = mapgen.create_world(map_seed=1, params=params)
        benchmark(world.tick)

    conf = {'mapgen.world_size': 128}
    ex.run(command_name='f', config_updates=conf)
