import numpy as np
import scipy.sparse as sp

from optimizers.abstract.Optimizer import Optimizer


class CG(Optimizer):
    def __init__(self, A, b, start_x,
                 iter_limit=None, tol=1e-8):

        super().__init__(start_x, bound=0,
                         iter_limit=iter_limit, tol=tol)
        self.A = A
        self.b = b

    def calculate_score(self):
        return np.linalg.norm(self.r)

    def init(self):
        self.r = self.b - self.A.dot(self.x)
        self.p = self.r
    
    def update(self):
        alpha = self.r.dot(self.r) / self.p.dot(self.A.dot(self.p))
        self.x = self.x + alpha * self.p
        r_next = self.r - alpha * self.A.dot(self.p)
        beta = r_next.dot(r_next) / self.r.dot(self.r)
        self.p = r_next + beta * self.p
        self.r = r_next

        
class PreconditionedCG(Optimizer):
    def __init__(self, A, b, start_x, M, solver=sp.linalg.spsolve,
                 iter_limit=None, tol=1e-8):

        super().__init__(start_x, bound=0,
                         iter_limit=iter_limit, tol=tol)
        self.A = A
        self.b = b
        self.M = M
        self.solver = solver

    def calculate_score(self):
        return np.linalg.norm(self.r)

    def init(self):
        self.r = self.b
        self.z = self.solver(self.M, self.b)
        self.p = self.z
    
    def update(self):
        alpha = self.r.dot(self.z) / self.p.dot(self.A.dot(self.p))
        self.x = self.x + alpha * self.p
        r_next = self.r - alpha * self.A.dot(self.p)
        z_next = self.solver(self.M, r_next)
        beta = r_next.dot(z_next) / self.r.dot(self.z)
        self.p = z_next + beta * self.p
        self.r = r_next   
        self.z = z_next