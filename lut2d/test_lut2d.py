import scipy.misc
import test_data
import lut2d
import numpy as np


def test_lut_simple():
    lut = np.zeros(2**9, dtype='uint8')
    lut[0] = 1
    res = np.zeros((64, 64), dtype='uint8')
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 1).all()
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 0).all()

    res = (scipy.misc.imread('test_input.png') > 0).astype('uint8')
    lut[:] = 1
    res = lut2d.binary_lut_filter(res, lut)
    assert (res == 1).all()


def test_lut_result():
    res = (scipy.misc.imread('test_input.png') > 0).astype('uint8')
    res = lut2d.binary_lut_filter(res, test_data.test_lut)
    scipy.misc.imsave('test_actual.png', res*255)
    expected = (scipy.misc.imread('test_expected.png') > 0).astype('uint8')
    assert (res == expected)[1:-1, 1:-1].all()  # test ignoring border conditions


def test_numba(benchmark):
    res = (scipy.misc.imread('test_input.png') > 0).astype('uint8')
    lut2d.binary_lut_filter(res, test_data.test_lut)  # compile aka warm-up
    benchmark(lut2d.binary_lut_filter, res, test_data.test_lut)
