import numpy as np
from collections import namedtuple

class CrossEntropyMethod:
    def __init__(self, mu0, sigma0, popsize, best_factor, rel_noise):
        self.mu = np.array(mu0)
        self.sigma = sigma0 * np.ones(shape=self.mu.shape)
        self.popsize = popsize
        self.best_factor = best_factor
        self.rel_noise = rel_noise

    def ask(self):
        population = self.mu + self.sigma * np.random.randn(self.popsize, self.mu.shape[0])
        return population

    def tell(self, population, losses):
        best_idx = np.argsort(losses)
        best_idx = best_idx[:int(self.best_factor * self.popsize)]
        self.mu = np.mean(population[best_idx], axis=0)
        self.sigma = np.std(population[best_idx], axis=0)
        self.sigma *= (1.0 + self.rel_noise)

        # reporting (cmaes compatibility)
        self.mean = self.mu
        Result = namedtuple('Result', 'xbest')
        self.result = Result(population[best_idx[0]])
        self.best_loss = min(losses)

    def disp(self):
        # cmaes compatibility
        print('Best loss:', self.best_loss)
        print('Mu min, mean, max: {:.6f}, {:.6f}, {:.6f}'.format(
            np.min(self.mu), np.mean(self.mu), np.max(self.mu)))
        print('Sigma min, mean, max: {:.6f}, {:.6f}, {:.6f}'.format(
            np.min(self.sigma), np.mean(self.sigma), np.max(self.sigma)))
