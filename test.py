#!/usr/bin/env python3
import sys
from subprocess import check_call
import imageio

check_call(['cmake', '-B', 'build', '.'])
check_call(['make', '-j3', '-C', 'build'])
sys.path.insert(0, './build')

from world import mapgen
size = 128
m = mapgen.Map(size)

for i in range(100):
    m.w.tick()
imageio.imwrite('test-output.png', m.render())
