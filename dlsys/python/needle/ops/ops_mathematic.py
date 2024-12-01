"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union, Dict

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy as np

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *
DTYPE = "float32"

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import te
from tvm import topi

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        print(node.inputs[0].shape, node.inputs[1].shape)
        print(node.inputs[0].dtype, node.inputs[1].dtype)
        A = node_map[node.inputs[0]]
        B = node_map[node.inputs[1]]
        
        def te_ewise_add(A, B):
            return topi.add(A, B)

        return bb.emit_te(te_ewise_add, A, B)
    
    def emit(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Expr], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]  # First input tensor
        B = node_map[node.inputs[1]]  # Second input tensor

        # Use Relax's add operator for element-wise addition
        add_expr = relax.op.add(A, B)

        # Emit the operation
        return bb.emit(add_expr)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        dx = y * power(x, add_scalar(y, -1))
        dy = power(x, y) * log(x)
        return (out_grad*dx, out_grad*dy)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        self.grad = self.scalar * (a ** self.scalar-1)
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * self.grad,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        self.grad_a = b ** -1
        self.grad_b = -a / (b**2)
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = out_grad * self.grad_a
        b = out_grad * self.grad_b
        return (a, b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        axes = [i for i in range(len(a.shape))]
        if self.axes is None: 
            axes[-1], axes[-2] = axes[-2], axes[-1]
        
        else:
            ax1, ax2 = self.axes
            axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
            
        return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION

    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
    
        A = node_map[node.inputs[0]]   
        axes = [i for i in range(len(node.inputs[0].shape))]

        if self.axes is None: 
            axes[-1], axes[-2] = axes[-2], axes[-1]
        
        else:
            ax1, ax2 = self.axes
            axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
        
        def te_transpose(A):
            return topi.transpose(A, axes=axes)

        return bb.emit_te(te_transpose, A)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION
    
    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]
        target_shape = self.shape 

        def te_reshape(A):
            return topi.reshape(A, target_shape)

        return bb.emit_te(te_reshape, A)
    
    def emit(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Expr], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]       # Input tensor to reshape
        target_shape = self.shape          # The target shape to reshape into

        # Use Relax's reshape operator
        reshape_expr = relax.op.reshape(A, target_shape)

        # Emit the operation
        return bb.emit(reshape_expr)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes_to_reduce = []
        orig_shape = node.inputs[0].shape
        broadcasted_shape = self.shape

        for i in range(len(broadcasted_shape)):
            if i >= len(orig_shape) or orig_shape[i] == 1:
                axes_to_reduce.append(i)
        res = summation(out_grad, tuple(axes_to_reduce))

        return reshape(res, orig_shape)
        ### END YOUR SOLUTION
    
    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]
        target_shape = self.shape  # Assuming self.shape contains the target shape

        def te_broadcast_to(A):
            return topi.broadcast_to(A, target_shape)

        return bb.emit_te(te_broadcast_to, A)


    def emit(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Expr], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]       # Input tensor to broadcast
        target_shape = self.shape          # The target shape to broadcast to

        # Use Relax's broadcast_to operator
        broadcast_expr = relax.op.broadcast_to(A, target_shape)

        # Emit the operation
        return bb.emit(broadcast_expr)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None: 
            self.axes = list(range(a.ndim))

        elif isinstance(self.axes, int):
            self.axes = [self.axes]        
    
        for ax in reversed(self.axes):
            a = array_api.sum(a.compact(), axis=ax, keepdims=False)

        return a.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)

        if self.axes is not None:
            # if not isinstance(self.axes, tuple): 
            #     self.axes = tuple(self.axes)
            for ax in self.axes:
                shape[ax] = 1
        
        else:
            shape = [1] * len(shape)
  
        return broadcast_to(reshape(out_grad, tuple(shape)), node.inputs[0].shape)
        ## END YOUR SOLUTION

    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        """
        Emit tensor expression for the summation operation.
        """
        A = node_map[node.inputs[0]]  # Input tensor

        # Determine axes
        axes = self.axes
        if axes is None:
            axes = list(range(len(A.shape)))

        # Define the TE function for summation
        def te_sum(A):
            return topi.sum(A, axis=axes, keepdims=False)

        # Emit the TE operation
        return bb.emit_te(te_sum, A)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        a = a.compact()
        b = b.compact()
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad @ transpose(b) # last two dims match a
        grad_b = transpose(a) @ out_grad # last two dims match b

        a_dims_to_sum = len(grad_a.shape) - len(a.shape)
        b_dims_to_sum = len(grad_b.shape) - len(b.shape)
        

        if a_dims_to_sum > 0:
            grad_a = summation(grad_a, tuple(range(a_dims_to_sum)))
            grad_a = reshape(grad_a, a.shape)

        if b_dims_to_sum > 0:
            grad_b = summation(grad_b, tuple(range(b_dims_to_sum)))
            grad_b = reshape(grad_b, b.shape)
        
        # print(f'matmul grad: {out_grad.dtype}')

        return (grad_a, grad_b)
        ### END YOUR SOLUTION

    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]
        B = node_map[node.inputs[1]]

        def te_matmul(A, B):
            return topi.matmul(A, B)

        return bb.emit_te(te_matmul, A, B)

    def emit(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Expr], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]  # First input tensor
        B = node_map[node.inputs[1]]  # Second input tensor

        # Use Relax's matmul operator
        matmul_expr = relax.op.matmul(A, B)

        # Emit the operation
        return bb.emit(matmul_expr)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1*a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1*out_grad
        ### END YOUR SOLUTION
      
    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
      raise NotImplementedError


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return divide(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0.0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(f'relu grad has nans: {np.isnan(out_grad.numpy()).any()}')
        # print(f'relu mask has nans: {np.isnan(self.mask.numpy()).any()}')
        res = out_grad * Tensor(node.inputs[0].cached_data > 0, requires_grad=False, device=node.inputs[0].device).detach()
        # print(f'relu propagates nans: {np.isnan(res.numpy()).any()}')
        return res
        ### END YOUR SOLUTION
    
    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]

        def te_relu(A):
            return topi.nn.relu(A)

        return bb.emit_te(te_relu, A)

    def emit(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Expr], node: Tensor) -> relax.Var:
        A = node_map[node.inputs[0]]  # Input tensor

        # Use Relax's ReLU operator
        relu_expr = relax.op.nn.relu(A)

        # Emit the operation
        return bb.emit(relu_expr)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.shape = a.shape
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (add_scalar(-1*tanh(node.inputs[0]) ** 2, 1))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        base_shape = list(args[0].shape)
        num_tensors = len(args)


        broadcast_shape = base_shape[:self.axis] + [1] + base_shape[self.axis:]
        combined_shape = base_shape[:self.axis] + [num_tensors] + base_shape[self.axis:]
        combined_array = (args[0] + 0).compact().reshape(broadcast_shape).broadcast_to(combined_shape).compact()
        combined_array.fill(0)
        
        for i, t in enumerate(args):
           
            bcast_tensor = t.compact().reshape(broadcast_shape)

            combined_array[(slice(None),) * self.axis + (i,) + (slice(None),) * (combined_array.ndim - self.axis - 1)] = bcast_tensor


        return combined_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res = []
        N = A.shape[self.axis]
        split_shape = [A.shape[i] for i in range(A.ndim) if i != self.axis]

        for i in range(N):
            t = (A[(slice(None),) * self.axis + (i,) + (slice(None),) * (A.ndim - self.axis - 1)])
            res.append(t.compact().reshape(split_shape))

        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
       
        new_shape = [a.shape[i] + (a.shape[i]) * self.dilation if i in self.axes else a.shape[i] for i in range(len(a.shape))]

        res = NDArray.make(new_shape, strides=None, device=a._device, handle=None, offset=0)
        res.fill(0.0)

        slices = [slice(None, None, self.dilation + 1) if i in self.axes else slice(None, None)
                for i in range(len(a.shape))]
     
        res[tuple(slices)] = a
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None, None, self.dilation + 1) if i in self.axes else slice(None)
              for i in range(len(a.shape))]
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        self.A = A.compact()
        self.B = B.compact()
        if self.padding > 0:
            pad_ax = [(0,0) if i not in (1, 2) else (self.padding, self.padding) for i in range(A.ndim)]
            A = A.pad(pad_ax)
            
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        
        Ho = (H - K) // self.stride + 1
        Wo = (W - K) // self.stride + 1

        inner_dim = K * K * C_in
        A = A.as_strided(shape = (N, Ho, Wo, K, K, C_in), strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact().reshape((N*Ho*Wo,inner_dim))
        out = A @ B.compact().reshape((K*K*C_in, C_out))
        return out.compact().reshape((N,Ho,Wo,C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        A, B = node.inputs

        # print(f'BACKWARD PASS \nA: {A.shape} B: {B.shape}\nstride: {self.stride} pad: {self.padding} out grad: {out_grad.shape}')

        # X.grad calculation
        w_flipped = transpose(flip(B, (0,1)), (2,3))
        out_grad_dilate = dilate(out_grad, (1, 2), self.stride - 1)

        N, H, W, Cin = out_grad_dilate.shape
        _, Ho, Wo, _ = A.shape
        K = B.shape[0]

        padding = (Ho - H + K - 1) // 2

        X_grad = conv(out_grad_dilate, w_flipped, padding=padding)
        
        # W.grad calculation
        A_transposed = transpose(A, (0, 3))  
        # 0,1,2,3 -> 2,1,0,3 -> 1,2,0,3
        out_grad_transposed = transpose(transpose(out_grad_dilate, (0, 2)), (0, 1))

        # convolve A_transposed with out_grad_transposed
        W_grad = conv(A_transposed, out_grad_transposed, padding=self.padding)
        
        # Transpose W_grad back to match the shape of B
        W_grad = transpose(transpose(W_grad, (0,1)), (1, 2)) #problem area

        # print(f'X: {A.shape} W: {B.shape} Xgrad: {X_grad.shape} Wgrad: {W_grad.shape}')
        return X_grad, W_grad
        
        ### END YOUR SOLUTION
  
    def emit_te(self, bb: relax.BlockBuilder, node_map: Dict[Tensor, relax.Var], node: Tensor) -> relax.Var:
        """
        Emit tensor expression for the conv2d operation.
        """
        print(f"conv2d: {node.inputs[0].shape} {node.inputs[1].shape}")
        A = node_map[node.inputs[0]]  # Input tensor
        B = node_map[node.inputs[1]]  # Filter tensor

        # Handle stride and padding
        stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        padding = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        dilation = (1, 1)  # Default dilation

        
        # Define the TE function
        def te_conv(A, B):
            return topi.nn.conv2d_nhwc(A, B, stride=stride, padding=padding, dilation=dilation)

        # Emit the TE operation
        return bb.emit_te(te_conv, A, B)


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


