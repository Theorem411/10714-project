from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z = Z.compact()
        max_Z = Z.max(axis=self.axes, keepdims=True)

        if self.axes is None: 
            max_Z = max_Z.compact().reshape(tuple([1]*Z.ndim))
        
        norm_Z = Z - max_Z.compact().broadcast_to(Z.shape)
        
        res = array_api.log(array_api.sum(array_api.exp(norm_Z), axis=self.axes, keepdims=False)) 
        max_Z_reshaped = array_api.reshape(max_Z.compact(), res.shape)

        res += max_Z_reshaped
        return res.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        sq_shape = list(node.inputs[0].shape)
        node_rs, out_grad_rs = node, out_grad

        if self.axes is not None:
            if not isinstance(self.axes, tuple): 
                self.axes = tuple([self.axes])
                
            for ax in self.axes:
                sq_shape[ax] = 1
            
            sq_shape = tuple(sq_shape)
            node_rs = broadcast_to(reshape(node, sq_shape), node.inputs[0].shape)
            out_grad_rs = broadcast_to(reshape(out_grad, sq_shape), node.inputs[0].shape)
        
        else:
            sq_shape = [1 for _ in range(len(sq_shape))]
            sq_shape = tuple(sq_shape)
            node_rs = broadcast_to(reshape(node, sq_shape), node.inputs[0].shape)
            out_grad_rs = broadcast_to(reshape(out_grad, sq_shape), node.inputs[0].shape)
        
        res = exp(node.inputs[0] - node_rs) * out_grad_rs
        return res
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)



