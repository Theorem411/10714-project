import tvm
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
import needle.nn as nn

from .model_eval import ModelEval

# Fix weight initialization: necessary for load_module to work
np.random.seed(4)

def ResidualConvBN(a1,b1,k1,s1,a2,b2,k2,s2, device=None, dtype="float32"):
    block = nn.Residual(nn.Sequential(
            nn.Conv(in_channels=a1, out_channels=b1, kernel_size=k1, stride=s1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b1, device=device),
            nn.ReLU(),
            nn.Conv(in_channels=a2, out_channels=b2, kernel_size=k2, stride=s2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b2, device=device),
            nn.ReLU()
        ))
    return block

def ConvBN(a1,b1,k1,s1,a2,b2,k2,s2, device=None, dtype="float32"):
    block = nn.Sequential(
            nn.Conv(in_channels=a1, out_channels=b1, kernel_size=k1, stride=s1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b1, device=device),
            nn.ReLU(),
            nn.Conv(in_channels=a2, out_channels=b2, kernel_size=k2, stride=s2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(b2, device=device),
            nn.ReLU()
        )
    return block
    

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(3,16,7,4,16,32,3,2,device,dtype),
            ResidualConvBN(32,32,3,1,32,32,3,1,device,dtype),
            ConvBN(32,64,3,2,64,128,3,2,device,dtype),
            ResidualConvBN(128,128,3,1,128,128,3,1,device,dtype),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype), 
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model.forward(x)
        ### END YOUR SOLUTION

################################################################################
# Performance eval wrapper class
################################################################################
class ConvEval(ModeEval):
  def __init__(self, input_dim, num_batches, batch_size,
               device="cpu", recompile=False):

    # input_dim, num_batches, batch_size not used
    super().__init__(input_dim, num_batches, batch_size, device, recompile)
  
  def module_lib_save_name(self):
    return f"conv-{self.ndl_device.name}.so"
  
  # !IMPORTANT
  def construct_model(self): 
    return ResNet9()

  # overrides default
  def dummy_input(self):
    return np.random.rand(self.batch_size, self.input_dim).astype(np.float32)