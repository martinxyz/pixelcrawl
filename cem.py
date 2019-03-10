import numpy as np
from collections import namedtuple

class CrossEntropyMethod:
    def __init__(self, mu0, sigma0, popsize, best_factor, mu_tau, cov_tau):
        self.mu = np.array(mu0)
        self.cov = sigma0 * np.eye(len(mu0))
        self.popsize = popsize
        self.best_factor = best_factor
        self.cov_tau = cov_tau
        self.mu_tau = mu_tau

    def ask(self):
        population = np.random.multivariate_normal(self.mu, self.cov, size=self.popsize)
        print('population.shape:', population.shape)
        return population

    def tell(self, population, losses):
        best_idx = np.argsort(losses)
        best_idx = best_idx[:int(self.best_factor * self.popsize)]

        mu = np.mean(population[best_idx], axis=0)
        cov = np.cov(population[best_idx], rowvar=False)

        mu_fac = np.exp(-1 / self.mu_tau) if self.mu_tau > 0 else 0
        cov_fac = np.exp(-1 / self.cov_tau) if self.cov_tau > 0 else 0

        self.mu = mu_fac * self.mu + (1 - mu_fac) * mu
        self.cov = cov_fac * self.cov + (1 - cov_fac) * cov

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
        # print('Sigma min, mean, max: {:.6f}, {:.6f}, {:.6f}'.format(
        #     np.min(self.sigma), np.mean(self.sigma), np.max(self.sigma)))
