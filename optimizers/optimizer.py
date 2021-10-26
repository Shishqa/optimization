class Optimizer:
    class State:
        def __init__(self, x, score):
            self.x = x
            self.score = score

    def __init__(self, f, best=State(0, 0),
                 iter_limit=None, no_improv_treshold=10e-5,
                 no_improv_iter_limit=10):

        self.iter_limit = iter_limit
        self.no_improv_treshold = no_improv_treshold
        self.no_improv_iter_limit = no_improv_iter_limit

        self.iters = 0
        self.no_improv_iters = 0
        self.best = best
        self.f = f

        self.history = {
            'x': [],
            'scores': [],
        }

    def update_history(self):
        self.history['x'].append(self.best.x)
        self.history['scores'].append(self.best.score)

    def update(self):
        pass

    def init(self):
        pass

    def get_best(self):
        pass

    def extra_terminate(self):
        return False

    def get_result(self):
        return self.best.x, self.best.score, self.history

    def terminate(self):
        if self.iter_limit and self.iters > self.iter_limit:
            print('reached iter limit')
            return True

        if self.no_improv_iters >= self.no_improv_iter_limit:
            print('no improvement after {} iterations'.format(self.iters))
            return True

        return self.extra_terminate()

    def update_best(self, new_best):
        prev_best = self.best
        if new_best.score < self.best.score:
            self.best = new_best
        return new_best.score < prev_best.score - self.no_improv_treshold

    def make_progress(self):
        new_best = self.get_best()

        self.iters += 1
        if not self.update_best(new_best):
            self.no_improv_iters += 1
        else:
            self.no_improv_iters = 0

        self.update_history()
        return not self.terminate()

    def optimize(self):
        self.iters = 0
        self.no_improv_iters = 0
        self.init()
        while self.make_progress():
            self.update()

        return self.get_result()

