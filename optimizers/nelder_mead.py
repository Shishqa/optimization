import copy
import numpy as np

from optimizer import Optimizer


class NelderMead(Optimizer):
    def __init__(self, f, start_x, step=0.1,
                 alpha=1., gamma=2., rho=0.5, sigma=0.5,
                 iter_limit=0, no_improv_treshold=10e-6,
                 no_improv_iter_limit=10):
        super().__init__(f, best=Optimizer.State(start_x, f(start_x)),
                         iter_limit=iter_limit,
                         no_improv_treshold=no_improv_treshold,
                         no_improv_iter_limit=no_improv_iter_limit)

        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        self.dim = 0
        self.start_x = start_x
        self.step = step

    @staticmethod
    def initial_simplex(f, dim, x_start, step):
        simplex = [Optimizer.State(x_start, f(x_start))]
        for i in range(dim):
            x = copy.copy(x_start)
            x[i] = x[i] + step
            simplex.append(Optimizer.State(x, f(x)))
        return simplex

    def init(self):
        self.dim = len(self.start_x)
        self.simplex = self.initial_simplex(self.f, self.dim, self.start_x,
                                            self.step)

    def get_best(self):
        self.simplex.sort(key=lambda p: p.score)
        return self.simplex[0]

    @staticmethod
    def centroid(dim, simplex):
        x0 = np.zeros(dim)
        for point in simplex[:-1]:
            x0 += point.x
        return x0 / dim

    def update(self):
        # centroid
        x0 = self.centroid(self.dim, self.simplex)

        # reflection
        xr = x0 + self.alpha * (x0 - self.simplex[-1].x)
        rscore = self.f(xr)
        if self.simplex[0].score <= rscore < self.simplex[-2].score:
            self.simplex[-1] = Optimizer.State(xr, rscore)
            return

        # expansion
        if rscore < self.simplex[0].score:
            xe = x0 + self.gamma * (x0 - self.simplex[-1].x)
            escore = self.f(xe)
            if escore < rscore:
                self.simplex[-1] = Optimizer.State(xe, escore)
                return
            else:
                self.simplex[-1] = Optimizer.State(xr, rscore)
                return

        # contraction
        xc = x0 + self.rho * (x0 - self.simplex[-1].x)
        cscore = self.f(xc)
        if cscore < self.simplex[-1].score:
            self.simplex[-1] = Optimizer.State(xc, cscore)
            return

        # shrinkage
        x1 = self.simplex[0].x
        new_simplex = []
        for p in self.simplex:
            redx = x1 + self.sigma * (p.x - x1)
            score = self.f(redx)
            new_simplex.append(Optimizer.State(redx, score))
        self.simplex = new_simplex
