from os.path import dirname, join
import numpy as np
import lut2d
import pixelcrawl
import time
from sacred import Ingredient
from functools import lru_cache

ing = Ingredient('mapgen')


@ing.config
def cfg():
    world_size = 128
    bias_fac = 0.1  # scale NN bias init (relative to weight init)
    agent_count = 200
    easy_start = False
    walk_through_wall_prob = 0.0625


@ing.capture
def count_params():
    ac = pixelcrawl.AgentController()
    cnt = 0
    cnt += np.prod(ac.w0.shape)
    cnt += np.prod(ac.b0.shape)
    cnt += np.prod(ac.w1.shape)
    cnt += np.prod(ac.b1.shape)
    return cnt


@lru_cache(maxsize=100)
def gen_walls_and_food(map_seed, size):
    rnd = np.random.RandomState(map_seed)
    lut_fn = join(dirname(__file__), 'blobgen_lut2d.dat')
    lut = np.loadtxt(lut_fn, dtype='uint8')

    walls = rnd.randint(0, 2, (size, size), dtype='uint8')
    for i in range(11):
        walls = lut2d.binary_lut_filter(walls, lut)

    food = rnd.randint(0, 2, (size, size), dtype='uint8')
    for i in range(9):
        food = lut2d.binary_lut_filter(food, lut)
        food[walls > 0] = 0

    return walls, food


@ing.capture
def create_world(map_seed, _seed, _run, world_size):
    world = pixelcrawl.World(_seed)
    walls, food = gen_walls_and_food(map_seed, world_size)
    world.init_map(walls, food)
    return world


@ing.capture
def add_agents(
    world, params, bias_fac, agent_count, easy_start, walk_through_wall_prob
):

    ac = pixelcrawl.AgentController()

    idx = [0]

    def randn(*shape):
        res = params[idx[0]:idx[0] + np.prod(shape)].reshape(*shape)
        idx[0] += np.prod(shape)
        assert shape == res.shape
        return res

    # something like Xavier and He initialization: https://stats.stackexchange.com/a/393012/52418
    # something like input normalization: * 2.1
    # note: w#.shape[1] counts the inputs, w#.shape[0] the outputs
    ac.w0 = randn(*ac.w0.shape) * np.sqrt(2 / ac.w0.shape[1]) * 2.1
    ac.b0 = randn(*ac.b0.shape) * bias_fac
    # note: followed by relu activation
    ac.w1 = randn(*ac.w1.shape) * np.sqrt(2 / (ac.w1.shape[1] + ac.w1.shape[0]))
    ac.b1 = randn(*ac.b1.shape) * bias_fac
    # note: followed by softmax activation

    if params is not None:
        assert idx[0] == len(params), idx

    world.init_agents(ac, agent_count, easy_start, walk_through_wall_prob)


@ing.capture
def render(world, world_size):
    img = np.zeros(shape=(world_size, world_size, 3), dtype='uint8')
    img[:, :, 0] = pixelcrawl.render_world(world, 0)
    img[:, :, 1] = pixelcrawl.render_world(world, 1)
    img[:, :, 2] = pixelcrawl.render_world(world, 2)
    return img
