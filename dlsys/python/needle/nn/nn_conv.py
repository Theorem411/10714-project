"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module



class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION

        receptive_field = kernel_size*kernel_size
        self.weight = Parameter(init.kaiming_uniform(in_channels*receptive_field, out_channels*receptive_field, shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, requires_grad=True, dtype=dtype))

        bias_interval = 1.0 / (in_channels*receptive_field)**0.5
        self.bias = Parameter(init.rand(out_channels, low=-bias_interval, high=bias_interval, device=device, requires_grad=True, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        def ceildiv(a, b):
            return (a + b - 1) // b


        padding = (self.kernel_size - 1) // 2

        res = ops.conv(self.convertToChannelLast(x), self.weight, stride=self.stride, padding=padding)
        res = self.convertToChannelFirst(res)

        if self.bias is not None:
            res += ops.broadcast_to(self.bias.reshape((1, self.out_channels, 1, 1)), res.shape)

        
        return res
        ### END YOUR SOLUTION

    def convertToChannelFirst(self, x):
        return ops.transpose(ops.transpose(x, (1,3)), (2,3))
    
    def convertToChannelLast(self, x):
        return ops.transpose(ops.transpose(x, (1,3)), (1,2))