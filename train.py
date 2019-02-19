#!/usr/bin/env python3
from world import mapgen
import numpy as np
import cma
import sys
import os
import imageio
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('cmaes-agent', ingredients=[mapgen.ing])
output_dir = None

@ex.config
def cfg(_log):
    cmaes_sigma = 0.4
    world_count = 5  # number of worlds to evaluate with
    world_ticks = 200
    evaluations = 50000  # maximum number of evaluations for this run
    render = None  # directory (or param filename) used by 'render' command
    use_eval_seed = False  # use a different map seed for each generation
    cmaes_popsize = None  # population size

# core loop (separated for easy profiling)
tick_callback = lambda world: None
def tick(world):
    world.tick()
    tick_callback(world)

@ex.capture
def evaluate(params, world_count, world_ticks, eval_seed=0):
    rewards = []
    for world_no in range(world_count):
        world = mapgen.create_world(map_seed=(world_no + eval_seed))
        mapgen.add_agents(world, params=params)
        for i in range(world_ticks):
            tick(world)
        rewards.append(world.total_score)
    mean_reward = np.mean(rewards)
    return mean_reward

def save_array(filename, data):
    with open(os.path.join(output_dir, filename), 'w') as f:
        np.savetxt(f, data)

@ex.command(unobserved=True)
def render(render):
    filename = render
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'xbest.dat')
    dirname = os.path.dirname(filename)
    global tick_callback
    i = [0]
    def tick_callback(world):
        i[0] += 1
        img = mapgen.render(world)
        fn = os.path.join(dirname, 'render-world-%06d.png' % i[0])
        imageio.imwrite(fn, img, compress_level=6)
    params = np.loadtxt(filename)
    reward = evaluate(params)
    print('reward:', reward)

@ex.main
def experiment_main(_run, _seed, cmaes_sigma, evaluations, use_eval_seed, cmaes_popsize):
    # while not es.stop():
    param_count = mapgen.count_params()
    print('param_count:', param_count)
    _run.info['param_count'] = param_count

    opts = {}
    if cmaes_popsize:
        opts['popsize'] = cmaes_popsize
    opts['seed'] = _seed
    es = cma.CMAEvolutionStrategy(param_count * [0], cmaes_sigma, opts)
    # logger = cma.CMADataLogger(os.path.join(output_dir, 'cmaes-')).register(es)

    evaluation = 0
    iteration = 0
    # while not es.stop():
    while evaluation < evaluations:
        solutions = es.ask()
        print('asked to evaluate', len(solutions), 'solutions')

        if use_eval_seed:
            eval_seed = np.random.randint(100000)
        else:
            eval_seed = 0

        rewards = [evaluate(x, eval_seed=eval_seed)
                   for x in solutions]
        # rewards = dask.compute(*rewards)
        evaluation += len(solutions)
        iteration += 1
        print('evaluation', evaluation)
        print('computed rewards:', list(reversed(sorted(rewards))))
        es.tell(solutions, [-r for r in rewards])

        _run.log_scalar("training.min_reward", min(rewards), evaluation)
        _run.log_scalar("training.max_reward", max(rewards), evaluation)
        _run.log_scalar("training.med_reward", np.median(rewards), evaluation)
        _run.log_scalar("training.avg_reward", np.average(rewards), evaluation)
        _run.result = max(rewards)

        save_array('xbest.dat', es.result.xbest)
        if iteration % 20 == 0:
            save_array(f'mean-eval%07d.dat' % evaluation, es.mean)
        # logger.add()  # write data to disc to be plotted
        es.disp()
    es.result_pretty()


def main():
    global output_dir
    args = sys.argv.copy()

    if '-h' in args or '--help' in args:
        ex.run_commandline([args[0], '--help'])
        sys.exit(0)

    if '-o' in args:
        idx = args.index('-o')
        args.pop(idx)
        output_dir = args.pop(idx)

        if os.path.exists(output_dir) and os.listdir(output_dir):
                print('Directory already has content! Not overwriting:', output_dir)
                sys.exit(1)
    else:
        output_dir = 'unnamed_output'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ex.observers.append(FileStorageObserver.create(output_dir))
    args.insert(1, '--name=' + os.path.split(output_dir)[-1])
    ex.run_commandline(args)

if __name__ == '__main__':
    main()
