from os.path import dirname, join
import numpy as np
import lut2d
import pixelcrawl
import time
from sacred import Ingredient

ing = Ingredient('mapgen')

@ing.config
def cfg():
    world_size = 128
    bias_fac = 0.1  # scale NN bias init (relative to weight init)
    l2_skew = 1.0

@ing.capture
def count_params(world_size):
    ac = pixelcrawl.AgentController()
    cnt = 0
    cnt += np.prod(ac.w0.shape)
    cnt += np.prod(ac.b0.shape)
    cnt += np.prod(ac.w1.shape)
    cnt += np.prod(ac.b1.shape)
    return cnt


@ing.capture
def create_world(params, map_seed,
                 _seed, _run, world_size, bias_fac, l2_skew):
    size = world_size
    rnd = np.random.RandomState(map_seed)

    lut_fn = join(dirname(__file__), 'blobgen_lut2d.dat')
    _run.add_artifact(lut_fn, metadata={'content-type': 'text/plain'})
    lut = np.loadtxt(lut_fn, dtype='uint8')

    walls = rnd.randint(0, 2, (size, size), dtype='uint8')
    for i in range(11):
        walls = lut2d.binary_lut_filter(walls, lut)

    food = rnd.randint(0, 2, (size, size), dtype='uint8')
    for i in range(9):
        food = lut2d.binary_lut_filter(food, lut)
        food[walls > 0] = 0

    w = pixelcrawl.World(_seed)
    w.init_map(walls, food)

    ac = pixelcrawl.AgentController()
    # weight_count = len(ac.w0) + len(ac.w1)
    # bias_count = len(ac.b0) + len(ac.b1)
    # agent_params = np.randn(weight_count + bias_count)
    # idx = 0

    if params is None:
        randn = np.random.randn
    else:
        idx = [0]
        def randn(*shape):
            res = params[idx[0]:idx[0]+np.prod(shape)].reshape(*shape)
            idx[0] += np.prod(shape)
            assert shape == res.shape
            return res

    ac.w0 = randn(*ac.w0.shape) / l2_skew
    ac.b0 = randn(*ac.b0.shape) * bias_fac / l2_skew
    ac.w1 = randn(*ac.w1.shape) * l2_skew
    ac.b1 = randn(*ac.b1.shape) * bias_fac * l2_skew

    if params is not None:
        assert idx[0] == len(params), idx

    w.init_agents(ac)

    return w

def render(world, world_size):
    img = np.zeros(shape=(world_size, world_size, 3), dtype='uint8')
    img[:, :, 0] = pixelcrawl.render_world(world, 0)
    img[:, :, 1] = pixelcrawl.render_world(world, 1)
    img[:, :, 2] = pixelcrawl.render_world(world, 2)
    return img
