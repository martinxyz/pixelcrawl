#!/usr/bin/env python3
from world import mapgen
import numpy as np
import cma
from sacred import Experiment
ex = Experiment('cmaes-agent', ingredients=[mapgen.ing])

@ex.config
def cfg(_log):
    cmaes_sigma = 0.4
    world_count = 5  # number of worlds to evaluate with
    iterations = 2000

@ex.capture
def evaluate(params, world_count):
    rewards = []
    for seed in range(world_count):
        world = mapgen.create_world(params=params, map_seed=seed)
        for i in range(200):
            world.tick()
        rewards.append(world.total_score)
    mean_reward = np.mean(rewards)
    return mean_reward

@ex.automain
def main(_run, cmaes_sigma, iterations):
    # while not es.stop():
    param_count = mapgen.count_params()
    print('param_count:', param_count)
    es = cma.CMAEvolutionStrategy(param_count * [0], cmaes_sigma)
    logger = cma.CMADataLogger('tmp/cmaes-').register(es)

    evaluations = 0
    
    # while not es.stop():
    for i in range(iterations):
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

    es.result_pretty()
