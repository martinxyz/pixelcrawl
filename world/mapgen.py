from os.path import dirname, join
import numpy as np
import lut2d
import pixelcrawl
import time

lut = np.loadtxt(join(dirname(__file__), 'blobgen_lut2d.dat'), dtype='uint8')

class Map:
    def __init__(self, size, seed=None):
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
        w.init_agents()

        self.w = w

    def render(self):
        img = np.zeros(shape=(self.size, self.size, 3), dtype='uint8')
        img[:, :, 0] = self.w.render(0)
        img[:, :, 1] = self.w.render(1)
        img[:, :, 2] = self.w.render(2)
        return img
