#!/usr/bin/env python3
import sys
from subprocess import check_call
import imageio
import numpy as np
import time

check_call(['cmake', '-B', 'build', '.'])
check_call(['make', '-j3', '-C', 'build'])
sys.path.insert(0, './build')

from world import mapgen
size = 128

world_count = 5

for seed in range(world_count):
    m = mapgen.Map(size, seed)
    t0 = time.time()
    for i in range(200):
        m.w.tick()
    print(time.time() - t0)  # 17ms
    print(m.w.total_score)

imageio.imwrite('test-output.png', m.render())
