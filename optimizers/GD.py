import types
import numpy as np

from optimizers.abstract.Optimizer import Optimizer


def backtracking (x, f, grad_f, rho, alpha0, beta1, beta2, tol=1e-17):
    
    alpha = alpha0
    if isinstance(grad_f, types.FunctionType):
        phi1 = f(x) -  beta1 * alpha * grad_f(x).dot(grad_f(x))
        phi2 = f(x) -  beta2 * alpha * grad_f(x).dot(grad_f(x))    
        f_k = f(x - alpha * grad_f(x)) 
    else:
        phi1 = f(x) -  beta1 * alpha * grad_f @ grad_f
        phi2 = f(x) -  beta2 * alpha * grad_f @ grad_f  
        f_k = f(x - alpha * grad_f) 

    while not ((f_k <= phi1) and (f_k >= phi2)):
        alpha *= rho
        if isinstance(grad_f, types.FunctionType):
            phi1 = f(x) - beta1 * alpha * grad_f(x).dot(grad_f(x))
            phi2 = f(x) - beta2 * alpha * grad_f(x).dot(grad_f(x))    
            f_k = f(x - alpha * grad_f(x)) 
        else: 
            phi1 = f(x) - beta1 * alpha * grad_f @ grad_f
            phi2 = f(x) - beta2 * alpha * grad_f @ grad_f  
            f_k = f(x - alpha * grad_f) 
        if alpha < tol:
            return alpha / rho

    return alpha


class GD(Optimizer):
    def __init__(self, f, grad_f, start_x,
                 rho=0.7, beta1=0.3,
                 iter_limit=None, tol=1e-8):

        super().__init__(start_x, bound=0,
                         iter_limit=iter_limit, tol=tol)

        self.f = f
        self.grad = grad_f

        self.rho = rho
        self.beta1 = beta1
        self.beta2 = 1.0 - beta1

    def calculate_score(self):
        return np.linalg.norm(self.grad(self.x))

    def update(self):
        alpha = backtracking(self.x, self.f, self.grad, 
                             rho=self.rho, alpha0=1., 
                             beta1=self.beta1, beta2=self.beta2)
        gradient = self.grad(self.x)
        self.x = self.x - alpha * gradient

        
class MomentumGD(Optimizer):
    def __init__(self, f, grad_f, start_x,
                 momentum=0.9, rho=0.7, beta1=0.3,
                 iter_limit=None, tol=1e-8):

        super().__init__(start_x, bound=0,
                         iter_limit=iter_limit, tol=tol)

        self.f = f
        self.grad = grad_f

        self.momentum = momentum
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = 1.0 - beta1

    def calculate_score(self):
        return np.linalg.norm(self.grad(self.x))

    def init(self):
        self.velocity = np.zeros(self.x.shape)
    
    def update(self):
        alpha = backtracking(self.x, self.f, self.grad, 
                             rho=self.rho, alpha0=1., 
                             beta1=self.beta1, beta2=self.beta2)
        gradient = self.grad(self.x)
        self.velocity = self.momentum * self.velocity - alpha * gradient
        self.x = self.x + self.velocity
     
    
class NesterovGD(Optimizer):
    def __init__(self, f, grad_f, start_x,
                 momentum=0.9, rho=0.7, beta1=0.3,
                 iter_limit=None, tol=1e-8):

        super().__init__(start_x, bound=0,
                         iter_limit=iter_limit, tol=tol)

        self.f = f
        self.grad = grad_f

        self.momentum = momentum
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = 1.0 - beta1

    def calculate_score(self):
        return np.linalg.norm(self.grad(self.x))

    def init(self):
        self.velocity = np.zeros(self.x.shape)
    
    def update(self):
        alpha = backtracking(self.x, self.f, self.grad, 
                             rho=self.rho, alpha0=1., 
                             beta1=self.beta1, beta2=self.beta2)
        
        proj = self.x + self.momentum * self.velocity
        gradient = self.grad(proj)
        
        self.velocity = self.momentum * self.velocity - alpha * gradient
        self.x = self.x + self.velocity