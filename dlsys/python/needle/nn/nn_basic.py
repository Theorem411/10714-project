"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features))) if bias else None
        # print(f'bias initialized as: {self.bias.shape}')
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = (X @ self.weight)

        if self.bias is None:
            return res
        
        broadcast_shape = (1,) * (len(X.shape) - 1) + (self.out_features,)
        # print(f'linear forward broadcasting bias: {self.bias.shape} -> {broadcast_shape} -> {res.shape}')
        res += ops.broadcast_to(self.bias.reshape(broadcast_shape), res.shape)
        ### END YOUR SOLUTION
        return res


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = list(X.shape)
        if len(shape) <= 2: return X

        new_dim = 1
        for i, s in enumerate(shape):
            new_dim *= s if i > 0 else 1

        return X.reshape((X.shape[0], new_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for mod in self.modules:
            x = mod.forward(x)
        return x 
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs = ops.logsumexp(logits, axes=1)
        rhs = ops.summation(init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype) * logits, axes=1)

        res = lhs - rhs
        res = ops.summation(res, axes=None) / Tensor(res.shape[0], dtype=res.dtype, device=logits.device, requires_grad=False)
        return res
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device

        self.weight = init.ones(dim, dtype=dtype, device=device)
        self.weight = Parameter(self.weight)
        self.bias = init.zeros(dim, dtype=dtype, device=device)
        self.bias = Parameter(self.bias)

        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)


        # debug num params: 
        # print(f"nn.BatchNorm1d: self.weight = {self.weight.shape} --> {np.prod(np.array(self.weight.shape))}")
        # print(f"nn.BatchNorm1d: self.bias = {self.bias.shape} --> {np.prod(np.array(self.bias.shape))}")

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        B, N = x.shape
        assert N == self.dim, "batchnorm input size does not match record"

        def running_update(xold, x):
          return (1 - self.momentum) * xold.data + self.momentum * x.data
        
        print(f"in BatchNorm1d: x.shape:{x.shape}")

        if self.training:

          # training: batch norm over features
          x_mean = ops.summation(x, axes=0) / B
          self.running_mean = running_update(self.running_mean, x_mean)
          x_mean = ops.broadcast_to(ops.reshape(x_mean, (1, N)), x.shape)

          x_minus_u = x - x_mean # ops.add(x, ops.negate(x_mean))
          x_var = ops.summation(x_minus_u ** 2, axes=0) / B # ops.divide_scalar(ops.summation(x_minus_u_2, axes=0), B)
          self.running_var = running_update(self.running_var, x_var)
          x_var = ops.broadcast_to(ops.reshape(x_var, (1, N)), x.shape)
          
          # # broadcast self.weight and self.bias
          w_broad = ops.broadcast_to(self.weight, x.shape)
          b_broad = ops.broadcast_to(self.bias, x.shape)

          # compute final result
          x_std = x_minus_u / ((x_var + self.eps) ** 0.5)# ops.power_scalar(ops.add_scalar(x_var, self.eps), 0.5) # ops.divide(x_minus_u, ops.power_scalar(ops.add_scalar(x_var, self.eps), 0.5))
          res = w_broad * x_std + b_broad # ops.add(ops.multiply(w_broad, x_std), b_broad)

          # print(f"BatchNorm 0 res.dtype: {res.dtype}")
          return res
        else: 
        
          x_minus_u = x - self.running_mean # ops.add(x, ops.negate(self.running_mean))

          x_std = x_minus_u / ((self.running_var + self.eps)**0.5) # ops.divide(x_minus_u, ops.power_scalar(ops.add_scalar(self.running_var, self.eps), 0.5))
          res = self.weight * x_std + self.bias

          return res
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.b = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E_x = x.sum(axes=1) / x.shape[1]
        Var_x =  ((x - E_x.reshape((E_x.shape[0], 1)).broadcast_to(x.shape)) ** 2).sum(axes=1) / x.shape[1] #E_xsq - E_x.__pow__(2) 

        E_x = E_x.reshape((E_x.shape[0], 1))
        Var_x = Var_x.reshape((Var_x.shape[0], 1))
     
        norm_term = ((x - E_x.broadcast_to(x.shape)) / (Var_x + self.eps).__pow__(0.5).broadcast_to(x.shape))
        y = self.w.reshape((1, self.dim)).broadcast_to(norm_term.shape) * norm_term + self.b.reshape((1, self.dim)).broadcast_to(norm_term.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            scale_factor = 1-self.p
        
        else:
            mask = init.randb(*x.shape, p=1.0) 
            scale_factor = 1.0
        
        return (x*mask) / scale_factor
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ff = self.fn(x)
        res = ff + Identity()(x)
        return res
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        print(f"_x shape: {_x.shape}")
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))

        res = y.transpose((2,3)).transpose((1,2))
        return res


