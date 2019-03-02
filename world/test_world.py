import numpy as np
import mapgen
from sacred import Experiment


def test_world_tick(benchmark):
    ex = Experiment('pytest', ingredients=[mapgen.ing])

    @ex.command
    def f():
        np.random.seed(1)
        params = np.random.randn(mapgen.count_params())
        world = mapgen.create_world(map_seed=1)
        mapgen.add_agents(world, params=params)
        benchmark(world.tick)

    conf = {'mapgen.world_size': 128}
    ex.run(command_name='f', config_updates=conf)
