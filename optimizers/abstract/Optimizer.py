class Optimizer:
    class State:
        def __init__(self, x, score):
            self.x = x
            self.score = score

    def __init__(self, start_x, bound=None,
                 iter_limit=None, tol=1e-8, 
                 no_improv_iter_limit=10):

        self.iter_limit = iter_limit
        self.tol = tol
        self.bound = bound
        self.no_improv_iter_limit = no_improv_iter_limit

        self.iters = 0
        self.no_improv_iters = 0

        self.x = start_x.copy()
        self.best = None

        self.history = {
            'x': [],
            'scores': [],
        }

    def update_history(self, x, score):
        self.history['x'].append(x)
        self.history['scores'].append(score)

    def calculate_score(self):
        pass

    def update(self):
        pass

    def init(self):
        pass

    def get_result(self):
        return self.best.x, self.best.score, self.history

    def extra_terminate(self):
        return False

    def terminate(self):
        if self.iter_limit and self.iters > self.iter_limit:
            print('reached limit of iterations')
            return True

        if self.bound is not None and self.best.score - self.bound < self.tol:
            return True
        elif self.bound is None and self.no_improv_iters >= self.no_improv_iter_limit:
            return True

        return self.extra_terminate()

    def update_best(self):
        new_score = self.calculate_score()

        if not self.best:
            self.best = Optimizer.State(self.x, new_score)
            return True

        prev_best = self.best
        if new_score < prev_best.score:
            self.best = Optimizer.State(self.x, new_score)

        return new_score < prev_best.score - self.tol

    def make_progress(self):
        self.iters += 1
        if not self.update_best():
            self.no_improv_iters += 1
        else:
            self.no_improv_iters = 0

        self.update_history(self.best.x, self.best.score)
        return not self.terminate()

    def optimize(self):
        self.iters = 0
        self.no_improv_iters = 0

        self.init()
        while self.make_progress():
            self.update()

        return self.get_result()

