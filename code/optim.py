import torch

#SGD optimizer
class SGD:
    def __init__(self, parameters, lr=0.001, momentum=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
        
    def step(self):
        for (i, p) in enumerate(self.parameters):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] + (1 - self.momentum) * p.grad
                p.data -= self.lr * self.v[i]

#Adam optimizer
class Adam:
    def __init__(self, parameters, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in parameters]
        self.v = [torch.zeros_like(p) for p in parameters]
        self.t = 0
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
    
    def step(self):
        self.t += 1
        for (i, p) in enumerate(self.parameters):
            if p.grad is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad**2
                m_hat = self.m[i] / (1 - self.betas[0]**self.t)
                v_hat = self.v[i] / (1 - self.betas[1]**self.t)
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)