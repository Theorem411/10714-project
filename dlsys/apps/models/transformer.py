import tvm
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
import needle.nn as nn

from .model_eval import ModelEval

################################################################################
# Performance eval wrapper class
################################################################################
class TransformerEval(ModelEval):
  def __init__(self, input_dim, num_batches, batch_size, 
               hidden_dim, seq_len, 
               n_layers=2, num_head=8, dim_head=32, dropout=0.0, causal=False, 
               device="cpu", recompile=False):
    super().__init__(input_dim, num_batches, batch_size, device, recompile)
    self.hidden_dim = hidden_dim
    self.seq_len = seq_len
    self.n_layers = n_layers
    self.num_head = num_head
    self.dim_head = dim_head
    self.dropout = dropout
    self.causal = causal

  def construct_model(self):
    return nn.Transformer(
        self.input_dim, self.hidden_dim, self.n_layers,
        num_head=self.num_head,
        dim_head=self.dim_head,
        dropout=self.dropout,
        causal=self.causal,
        sequence_len=self.seq_len,
        device=self.ndl_device,
        batch_first=True,
    )

  # overrides default
  def dummy_input(self):
    return np.random.randint(low=0, high=self.seq_len, size=(self.seq_len, self.batch_size, self.input_dim)).astype(np.float32)

  # overrides default
  def module_lib_save_name(self):
    return f"trans-{self.ndl_device.name}.so"