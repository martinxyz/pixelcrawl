import numpy as np
import mapgen

def test_world_tick(benchmark):
    param_count = 590
    np.random.seed(1)
    params = np.random.randn(param_count)

    m = mapgen.Map(size=128, seed=1, params=params)
    benchmark(m.w.tick)
