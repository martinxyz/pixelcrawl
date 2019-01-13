from os.path import dirname, join
import numpy as np
import lut2d

lut = np.loadtxt(join(dirname(__file__), 'blobgen_lut2d.dat'), dtype='uint8')

class Map:
    def __init__(self, size, seed=None):
        self.size = size
        rnd = np.random.RandomState(seed)

        self.walls = rnd.randint(0, 2, (size, size), dtype='uint8')
        for i in range(11):
            self.walls = lut2d.binary_lut_filter(self.walls, lut)

        self.food = rnd.randint(0, 2, (size, size), dtype='uint8')
        for i in range(9):
            self.food = lut2d.binary_lut_filter(self.food, lut)
            self.food[self.walls > 0] = 0

    def render(self):
        img = np.zeros(shape=(self.size, self.size, 3))

        def add_layer(bitmap, color, alpha=1.0):
            img[bitmap > 0, :] *= 1 - alpha
            img[bitmap > 0, 0] += alpha * color[0]
            img[bitmap > 0, 1] += alpha * color[1]
            img[bitmap > 0, 2] += alpha * color[2]

        add_layer(self.walls, (255, 255, 255))
        add_layer(self.food, (200, 80, 80), 0.8)
        return img.astype('uint8')


if __name__ == '__main__':
    import imageio
    size = 128
    map = Map(size)
    imageio.imwrite('mapgen-output.png', map.render())
