from collections.abc import Iterable


class SGD:
    def __init__(
        self,
        params: Iterable[Parameter],
        lr=1e-3
    ):
        self.params = params
        self.lr = lr


    def zero_grad(self):
        for p in self.params:
            p.grad = None
    

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.weights -= self.lr * p.grad
