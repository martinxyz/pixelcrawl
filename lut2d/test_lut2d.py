import imageio
import lut2d
import os
import numpy as np
from . import test_data

def load_img(fn):
    fn = os.path.join(os.path.dirname(__file__), fn)
    return (imageio.imread(fn) > 0).astype('uint8')

def test_lut_simple():
    lut = np.zeros(2**9, dtype='uint8')
    lut[0] = 1
    res = np.zeros((64, 64), dtype='uint8')
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 1).all()
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 0).all()

    res = load_img('test_input.png')
    lut[:] = 1
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 1).all()


def test_lut_result():
    res = load_img('test_input.png')
    res = lut2d.binary_lut_filter(res, test_data.test_lut)
    # imageio.imwrite('test_actual.png', res*255)

    expected = load_img('test_expected.png')
    assert (res == expected)[1:-1, 1:-1].all()  # test ignoring border conditions


def test_numba(benchmark):
    res = load_img('test_input.png')
    lut2d.binary_lut_filter(res, test_data.test_lut)  # compile aka warm-up
    benchmark(lut2d.binary_lut_filter, res, test_data.test_lut)
