from os.path import dirname, join
import numpy as np
import lut2d
import pixelcrawl
import time

lut = np.loadtxt(join(dirname(__file__), 'blobgen_lut2d.dat'), dtype='uint8')

class Map:
    def __init__(self, size, params, seed=None):
        w = pixelcrawl.World()

        if seed is None:
            seed = int(time.time() * 1000) % 2**30
        self.size = size
        rnd = np.random.RandomState(seed)
        w.seed(seed)

        walls = rnd.randint(0, 2, (size, size), dtype='uint8')
        for i in range(11):
            walls = lut2d.binary_lut_filter(walls, lut)

        food = rnd.randint(0, 2, (size, size), dtype='uint8')
        for i in range(9):
            food = lut2d.binary_lut_filter(food, lut)
            food[walls > 0] = 0

        w.init_map(walls, food)

        # a = pixelcrawl.Agent()
        # a.x = 4
        ac = pixelcrawl.AgentController()
        # weight_count = len(ac.w0) + len(ac.w1)
        # bias_count = len(ac.b0) + len(ac.b1)
        # agent_params = np.randn(weight_count + bias_count)
        # idx = 0

        # randn = np.random.randn
        idx = [0]
        def randn(*shape):
            res = params[idx[0]:idx[0]+np.prod(shape)].reshape(*shape)
            idx[0] += np.prod(shape)
            assert shape == res.shape
            return res
        ac.w0 = randn(*ac.w0.shape)
        ac.b0 = randn(*ac.b0.shape) * 0.1
        ac.w1 = randn(*ac.w1.shape)
        ac.b1 = randn(*ac.b1.shape) * 0.1
        assert idx[0] == len(params), idx

        w.init_agents(ac)

        self.w = w

    def render(self):
        img = np.zeros(shape=(self.size, self.size, 3), dtype='uint8')
        img[:, :, 0] = pixelcrawl.render_world(self.w, 0)
        img[:, :, 1] = pixelcrawl.render_world(self.w, 1)
        img[:, :, 2] = pixelcrawl.render_world(self.w, 2)
        return img
