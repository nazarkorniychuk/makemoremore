"""
Optimization Algorithms Implementation

This module provides implementations of common optimization algorithms
for training neural networks, including:
- Stochastic Gradient Descent (SGD) with momentum
- Adam optimizer with adaptive learning rates

All optimizers follow a similar interface with zero_grad() and step() methods.

Author: Nazar Korniichuk
Date: 02.07.2025
"""

import torch

#SGD optimizer
class SGD:
    """
    Stochastic Gradient Descent optimizer with optional momentum.
    
    Implements the classic SGD algorithm with optional momentum for
    better convergence in some cases.
    
    Args:
        parameters (list): List of parameter tensors to optimize
        lr (float): Learning rate (default: 0.001)
        momentum (float): Momentum factor (default: 0)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, parameters, lr=0.001, momentum=0, device = 'cpu'):
        self.device = device
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p, device=device) for p in parameters]
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for p in self.parameters:
            p.grad = None
    def to(self, device):
        self.device = device
        self.v = [v.to(device) for v in self.v]
        return self
        
    def step(self):
        """
        Perform a single optimization step.
        
        Updates parameters using the current gradients and momentum.
        """
        for (i, p) in enumerate(self.parameters):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] + (1 - self.momentum) * p.grad
                p.data -= self.lr * self.v[i]

#Adam optimizer
class Adam:
    """
    Adam optimizer with adaptive learning rates.
    
    Implements the Adam algorithm which adapts learning rates for each
    parameter based on estimates of first and second moments of gradients.
    
    Args:
        parameters (list): List of parameter tensors to optimize
        lr (float): Learning rate (default: 0.001)
        betas (tuple): Coefficients for computing running averages (default: (0.9, 0.999))
        eps (float): Small constant for numerical stability (default: 1e-8)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, parameters, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8, device = 'cpu'):
        self.device = device
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [torch.zeros_like(p, device = device) for p in parameters]
        self.v = [torch.zeros_like(p, device = device) for p in parameters]
        self.t = 0
        
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()  # More efficient than setting to None
    def to(self, device):
        self.device = device
        self.m = [p.to(device) for p in self.m]
        self.v = [p.to(device) for p in self.v]
        return self
    
    def step(self):
        """
        Perform a single optimization step.
        
        Updates parameters using Adam algorithm with bias correction.
        """
        self.t += 1
        for (i, p) in enumerate(self.parameters):
            if p.grad is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad**2
                m_hat = self.m[i] / (1 - self.betas[0]**self.t)
                v_hat = self.v[i] / (1 - self.betas[1]**self.t)
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)