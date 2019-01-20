#!/usr/bin/env python3
import sys
from subprocess import check_call
import numpy as np
import cma

check_call(['cmake', '-B', 'build', '.'])
check_call(['make', '-j3', '-C', 'build'])
sys.path.insert(0, './build')

from world import mapgen
size = 128

world_count = 5
param_count = 590

def evaluate(params):
    rewards = []
    for seed in range(world_count):
        m = mapgen.Map(size, params, seed)
        for i in range(200):
            m.w.tick()
        rewards.append(m.w.total_score)
    mean_reward = np.mean(rewards)
    return mean_reward


es = cma.CMAEvolutionStrategy(param_count * [0], 0.4)
logger = cma.CMADataLogger('cmaes-').register(es)

while not es.stop():
    solutions = es.ask()
    print('asked to evaluate', len(solutions), 'solutions')

    rewards = [evaluate(x) for x in solutions]
    # rewards = dask.compute(*rewards)
    print('computed rewards:', rewards)
    es.tell(solutions, [-r for r in rewards])

    logger.add()  # write data to disc to be plotted
    es.disp()
