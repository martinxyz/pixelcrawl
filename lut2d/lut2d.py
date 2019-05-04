from numba import guvectorize
import numpy as np

"""Filter a 2d binary image using a 3x3 neighbourhood LUT.

Using periodic boundary conditions.
Input pixels are assumed to be {0, 1} of type uint8.
The 9-bit LUT is given as uint8 array of length 2^9.
"""
def binary_lut_filter(inp, lut):
    assert "{0:b}".format(inp.shape[0]).count('1') == 1, 'shape must be a power of two'
    assert "{0:b}".format(inp.shape[1]).count('1') == 1, 'shape must be a power of two'
    assert inp.shape[0] == inp.shape[1], 'input must be square'  # to use a single mask
    assert len(lut) == 2**9, 'LUT must have 2^9 entries'
    result = np.zeros_like(inp)
    binary_lut_filter_inner(inp, lut, result)
    # print(binary_lut_filter_inner.inspect_types())
    return result


# measured to be just 1.5x slower than the same written C
@guvectorize(["void(uint8[:,:], uint8[:], uint8[:,:])"],
             "(n,n),(k)->(n,n)", nopython=True)
def binary_lut_filter_inner(src, lut, result):
    # everything except borders
    h = src.shape[0]
    w = src.shape[1]
    for y in range(1, h-1):
        for x in range(1, w-1):
            key = 0
            key |= (src[y-1, x-1] & 1) << 0
            key |= (src[y-1, x+0] & 1) << 1
            key |= (src[y-1, x+1] & 1) << 2
            key |= (src[y+0, x-1] & 1) << 3
            key |= (src[y+0, x+0] & 1) << 4
            key |= (src[y+0, x+1] & 1) << 5
            key |= (src[y+1, x-1] & 1) << 6
            key |= (src[y+1, x+0] & 1) << 7
            key |= (src[y+1, x+1] & 1) << 8
            result[y, x] = lut[key]

    # borders
    MASK = w-1  # w == h; always a power of two (assertion above)
    for y in range(-1, 1):
        for x in range(w):
            key = 0
            key |= (src[(y-1)&MASK, (x-1)&MASK] & 1) << 0
            key |= (src[(y-1)&MASK, (x+0)&MASK] & 1) << 1
            key |= (src[(y-1)&MASK, (x+1)&MASK] & 1) << 2
            key |= (src[(y+0)&MASK, (x-1)&MASK] & 1) << 3
            key |= (src[(y+0)&MASK, (x+0)&MASK] & 1) << 4
            key |= (src[(y+0)&MASK, (x+1)&MASK] & 1) << 5
            key |= (src[(y+1)&MASK, (x-1)&MASK] & 1) << 6
            key |= (src[(y+1)&MASK, (x+0)&MASK] & 1) << 7
            key |= (src[(y+1)&MASK, (x+1)&MASK] & 1) << 8
            result[y&MASK, x&MASK] = lut[key]

    for y in range(h):
        for x in range(-1, 1):
            key = 0
            key |= (src[((y-1)&MASK, (x-1)&MASK)] & 1) << 0
            key |= (src[((y-1)&MASK, (x+0)&MASK)] & 1) << 1
            key |= (src[((y-1)&MASK, (x+1)&MASK)] & 1) << 2
            key |= (src[((y+0)&MASK, (x-1)&MASK)] & 1) << 3
            key |= (src[((y+0)&MASK, (x+0)&MASK)] & 1) << 4
            key |= (src[((y+0)&MASK, (x+1)&MASK)] & 1) << 5
            key |= (src[((y+1)&MASK, (x-1)&MASK)] & 1) << 6
            key |= (src[((y+1)&MASK, (x+0)&MASK)] & 1) << 7
            key |= (src[((y+1)&MASK, (x+1)&MASK)] & 1) << 8
            result[y&MASK, x&MASK] = lut[key]

