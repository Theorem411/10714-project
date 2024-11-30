import numpy as np
import tvm
from tvm import relax, transform, meta_schedule
from tvm import ir

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
import needle.nn as nn

class ModelEval:
  model: nn.Module

  def __init__(): 
    raise NotImplementedError

  @property
  def model():
    return model

  def dummy_input():
    raise NotImplementedError