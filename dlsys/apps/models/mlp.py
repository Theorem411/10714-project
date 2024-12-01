import tvm
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
import needle.nn as nn

from .model_eval import ModelEval

# Fix Linear weight initialization: necessary for load_module to work
np.random.seed(4)

class MLPModel(nn.Module):
    def __init__(self, input_dim=512, n_layers=3, device=None):
            components = []
            for _ in range(n_layers):
                components.extend([
                  nn.Linear(input_dim, input_dim, bias=True, device=device), 
                  nn.ReLU()
                ])
            self.mlp = nn.Sequential(*components)
    def forward(self, x):
        return self.mlp(x)

################################################################################
# Performance eval wrapper class
################################################################################
class MLPEval(ModelEval):
  def __init__(self, input_dim, num_batches, batch_size, n_layers,
               device="cpu", recompile=False):

    self.n_layers = n_layers
    super().__init__(input_dim, num_batches, batch_size, device, recompile)
  
  def module_lib_save_name(self):
    return f"mlp-{self.ndl_device.name}.so"
  
  # !IMPORTANT
  def construct_model(self): 
    return MLPModel(self.input_dim, self.n_layers, self.ndl_device)

  # overrides default
  def dummy_input(self):
    return np.random.rand(self.batch_size, self.input_dim).astype(np.float32)