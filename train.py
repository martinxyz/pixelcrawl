#!/usr/bin/env python3
from world import mapgen
import numpy as np
import cma
import sys
import os
import imageio
from sacred import Experiment
from sacred.observers import FileStorageObserver
import dask
import dask.multiprocessing

sys.path.insert(0, 'external/fCMApy')
from fCSA import fCSA
from CSA import CSA

ex = Experiment('cmaes-agent', ingredients=[mapgen.ing])
output_dir = None


@ex.config
def cfg(_log):
    cmaes_sigma = 1.0
    world_count = 5  # number of worlds to evaluate with
    world_ticks = 200
    evaluations = 50000  # maximum number of evaluations for this run
    render = None  # directory (or param filename) used by 'render' command
    use_eval_seed = False  # use a different map seed for each generation
    cmaes_popsize = None  # population size
    algo = 'cma'  # cma | csa | fcsa
    fcsa_noise_adaptation = True


# core loop (separated for easy profiling)
def tick_callback(world):
    pass


def tick(world):
    world.tick()
    tick_callback(world)


@ex.capture
def evaluate(params, world_count, world_ticks, eval_seed=0):
    # XXX: How does multiprocessing interact with sacred's predicable rng
    #      seeds? Ideally all seeds are derived from parameters of the
    #      dask.delayed and then we're done. Not caring much about
    #      predictability, but currently create_world() also uses sacred's
    #      _seed feature which is supposed to generate a different seed per
    #      call. Maybe now a different process can get the same _seed?

    def eval_world(world_no):
        world = mapgen.create_world(map_seed=(world_no + eval_seed))
        mapgen.add_agents(world, params=params)
        for i in range(world_ticks):
            tick(world)
        return world.total_score

    # Simplify: use ProcessPoolExecutor.map() instead.
    # ref: https://github.com/zuoxingdong/lagom/blob/master/baselines/cem/experiment.py#L108

    rewards = [dask.delayed(eval_world)(world_no)
               for world_no in range(world_count)]
    return dask.delayed(np.mean)(rewards)


def save_array(filename, data):
    with open(os.path.join(output_dir, filename), 'w') as f:
        np.savetxt(f, data)


@ex.command(unobserved=True)
def render(render):
    dask.config.set(scheduler='synchronous')
    filename = render
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'xbest.dat')
    dirname = os.path.dirname(filename)
    global tick_callback
    i = [0]

    def tick_callback(world):
        i[0] += 1  # XXX this is broken, order is not consistent
        img = mapgen.render(world)
        fn = os.path.join(dirname, 'render-world-%06d.png' % i[0])
        imageio.imwrite(fn, img, compress_level=6)

    params = np.loadtxt(filename)
    reward = evaluate(params)
    reward = dask.compute(reward)
    print('reward:', reward)


@ex.main
def experiment_main(
    _run, _seed, cmaes_sigma, evaluations, use_eval_seed, cmaes_popsize, algo, fcsa_noise_adaptation
):
    param_count = mapgen.count_params()
    print('param_count:', param_count)
    _run.info['param_count'] = param_count

    if algo == 'cma':
        opts = {}
        if cmaes_popsize:
            opts['popsize'] = cmaes_popsize
        opts['seed'] = _seed
        es = cma.CMAEvolutionStrategy(param_count * [0], cmaes_sigma, opts)
    elif algo == 'fcsa':
        es = fCSA(np.zeros(param_count), cmaes_sigma**2,
                  noise_adaptation=fcsa_noise_adaptation,
                  popsize=cmaes_popsize)
    elif algo == 'csa':
        es = CSA(np.zeros(param_count), cmaes_sigma**2, popsize=cmaes_popsize)
    else:
        print('algo', repr(algo), 'not implemented')
        sys.exit(1)

    evaluation = 0
    iteration = 0
    rewards_logging = []
    while evaluation < evaluations:
        solutions = es.ask()
        print('asked to evaluate', len(solutions), 'solutions')

        if use_eval_seed:
            eval_seed = np.random.randint(100_000)
        else:
            eval_seed = 0

        rewards = [evaluate(x, eval_seed=eval_seed) for x in solutions]
        rewards = dask.compute(*rewards)
        evaluation += len(solutions)
        iteration += 1
        print('evaluation', evaluation)
        print('computed rewards:', list(reversed(sorted(rewards))))
        es.tell(solutions, [-r for r in rewards])

        rewards_logging.extend(rewards)
        # save_array('xbest.dat', es.result.xbest)
        if iteration % 10 == 0:
            _run.log_scalar("training.min_reward", min(rewards_logging), evaluation)
            _run.log_scalar("training.max_reward", max(rewards_logging), evaluation)
            _run.log_scalar("training.med_reward", np.median(rewards_logging), evaluation)
            _run.log_scalar("training.avg_reward", np.average(rewards_logging), evaluation)
            _run.log_scalar("training.iteration", iteration, evaluation)
            _run.result = max(rewards_logging)
            rewards_logging = []

            if algo in ['csa', 'fcsa']:
                _run.log_scalar("algo.variance", es.variance, evaluation)
                _run.log_scalar("algo.n_off", es.n_off, evaluation)
                _run.log_scalar("algo.mu_eff", es._mu_eff, evaluation)
            if algo == 'fcsa':
                _run.log_scalar("algo.sigma_noise", es._sigma_noise, evaluation)

            mean = es.result.xfavorite if algo == 'cma' else es.mean
            best = es.result.xbest if algo == 'cma' else mean

            _run.log_scalar("training.xmscale", np.sqrt(np.mean(mean**2)), evaluation)

            if iteration % 100 == 0:
                save_array('xbest.dat', best)
                save_array(f'xfavorite-eval%07d.dat' % evaluation, mean)

            if algo == 'cma':
                save_array('xbest.dat', es.mean)
                save_array(f'xfavorite-eval%07d.dat' % evaluation, es.result.xfavorite)
                save_array(f'stds-eval%07d.dat' % evaluation, es.result.stds)
                es.disp()


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

    dask.config.set(scheduler='processes')
    # dask.config.set(scheduler='synchronous')
    ex.run_commandline(args)


if __name__ == '__main__':
    main()
