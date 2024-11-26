"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            grad = p.grad.detach()
            grad += self.weight_decay * p.data

            prev_u = self.u[p] if p in self.u else ndl.init.zeros(*p.data.shape, requires_grad=False, dtype=p.data.dtype, device=p.device).detach()
            self.u[p] = self.momentum * prev_u + (1-self.momentum)*grad

            p.data -= self.lr * self.u[p]

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            grad = p.grad.detach()
            grad += self.weight_decay * p.data

            prev_m = self.m[p] if p in self.m else ndl.init.zeros(*p.data.shape, requires_grad=False, device=p.device, dtype=p.data.dtype).detach()
            self.m[p] = self.beta1 * prev_m + (1 - self.beta1)*grad

            prev_v = self.v[p] if p in self.v else ndl.init.zeros(*p.data.shape, requires_grad=False, device=p.device, dtype=p.data.dtype).detach()
            self.v[p] = self.beta2 * prev_v + (1 - self.beta2)*grad.__pow__(2)

            m_hat = self.m[p] / (1 - (self.beta1**self.t))
            v_hat = self.v[p] / (1 - (self.beta2**self.t))
            
            # print(f'p data in: {p.data}')
            # print(f'lr: {self.lr} mhat: {m_hat} vhat: {v_hat}')
            p.data -= self.lr * (m_hat / ((v_hat).__pow__(0.5) + self.eps))
        

        ### END YOUR SOLUTION
