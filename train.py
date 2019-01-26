#!/usr/bin/env python3
from world import mapgen
from experiment import ex
import numpy as np
import cma

@ex.capture
def evaluate(params, world_count, world_size):
    rewards = []
    for seed in range(world_count):
        m = mapgen.Map(world_size, params, seed)
        for i in range(200):
            m.w.tick()
        rewards.append(m.w.total_score)
    mean_reward = np.mean(rewards)
    return mean_reward

@ex.automain
def main(_run, cmaes_sigma):
    # while not es.stop():
    param_count = 590
    es = cma.CMAEvolutionStrategy(param_count * [0], cmaes_sigma)
    logger = cma.CMADataLogger('outputs/pix3/cmaes-').register(es)

    evaluations = 0
    
    # while not es.stop():
    for i in range(2000):
        solutions = es.ask()
        print('asked to evaluate', len(solutions), 'solutions')

        rewards = [evaluate(x) for x in solutions]
        # rewards = dask.compute(*rewards)
        print('computed rewards:', rewards)
        es.tell(solutions, [-r for r in rewards])

        evaluations += len(rewards)
        _run.log_scalar("training.min_reward", min(rewards), evaluations)
        _run.log_scalar("training.max_reward", max(rewards), evaluations)
        _run.log_scalar("training.med_reward", np.median(rewards), evaluations)
        _run.log_scalar("training.avg_reward", np.average(rewards), evaluations)

        logger.add()  # write data to disc to be plotted
        es.disp()
