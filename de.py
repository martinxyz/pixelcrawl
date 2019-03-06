import numpy as np
from collections import namedtuple

# slightly based on: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python
class DifferentialEvolution:
    def __init__(self, mu0, sigma0, popsize, mut, crossp):
        mu = np.array(mu0)
        sigma = sigma0 * np.ones(shape=mu.shape)
        self.pop0 = mu + sigma * np.random.randn(popsize, mu.shape[0])
        self.pop = None
        self.mut = mut
        self.crossp = crossp

    def ask(self):
        if self.pop is None:
            print('initial pop')
            return self.pop0

        popsize = len(self.pop)
        dimensions = len(self.pop[0])

        best_idx = np.argsort(self.pop_losses)

        trials = []
        for j in range(len(self.pop)):
            # idxs = [idx for idx in range(popsize) if idx != j]
            # a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            idxs = [idx for idx in range(popsize) if idx != j and idx != best_idx[0]]
            a = self.pop[best_idx[0]]
            b, c = self.pop[np.random.choice(idxs, 2, replace=False)]
            mutant = a + self.mut * (b - c)
            cross_points = np.random.rand(dimensions) < self.crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trials.append(np.where(cross_points, mutant, self.pop[j]))
            if j == 0:
                print('mut', a[0], b[0], c[0], '-->', mutant[0], '-->', trials[-1][0])
        self.trials = np.array(trials)
        trials.extend(self.pop)
        return trials

    def tell(self, trials, losses):
        if self.pop is None:
            self.pop = trials
            self.pop_losses = losses
        else:
            popsize = len(self.pop)
            self.pop_losses = losses[popsize:]  # update re-evaluated

            trials = trials[:popsize]
            losses = losses[:popsize]
            print('trials[0]', trials[0])
            print('losses[0]', losses[0])
            trials = np.array(trials)
            assert (trials == self.trials).all()
            for j in range(popsize):
                loss = losses[j]
                if loss < self.pop_losses[j]:
                    print('replace loss', self.pop_losses[j], 'with', loss)
                    self.pop_losses[j] = loss
                    self.pop[j] = trials[j]
                else:
                    print('keep loss', self.pop_losses[j], 'instead of new', loss)

        # note: no elitism here; best of trial, not best super-lucky individuum ever
        best_idx = np.argsort(losses)
        Result = namedtuple('Result', 'xbest')
        self.result = Result(trials[best_idx[0]])
        self.best_loss = min(losses)

    def disp(self):
        # cmaes compatibility
        print('Best loss:', self.best_loss)
