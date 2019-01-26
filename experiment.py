from sacred import Experiment
ex = Experiment('cmaes-agent')

@ex.config
def cfg(_log):
    world_size = 128
    world_count = 5
    cmaes_sigma = 0.4
    bias_fac = 0.1
    l2_skew = 1.0
