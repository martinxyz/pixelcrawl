#!/usr/bin/env python3
from world import mapgen
import imageio
import time


size = 128

world_count = 5

for seed in range(world_count):
    m = mapgen.Map(size, seed=seed)
    t0 = time.time()
    for i in range(200):
        m.w.tick()
    print(time.time() - t0)  # 17ms
    print(m.w.total_score)

imageio.imwrite('test-output.png', m.render())
