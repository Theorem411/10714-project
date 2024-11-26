import sys
sys.path.append('./python')
sys.path.append('./apps')
import itertools
import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd

import tvm
from tvm import relax

from models import *
from mlp import MLPModel

np.random.seed(0)

_DEVICE = [ndl.cpu(), pytest.pararm(ndl.cuda(), )]