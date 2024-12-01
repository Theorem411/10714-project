import argparse
import numpy as np
import tvm
from tvm import relax, transform, meta_schedule
from tvm.contrib import utils
from tvm import ir

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
from needle_tvm import to_tvm_tensor, to_tvm_fx

import needle.nn as nn
from models import MLPEval, ConvEval, TransformerEval
from utils import *

np.random.seed(0)

def getoptions():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-r", "--recompile",
    action="store_true",
    help="recompile tvm module",
  )
  parser.add_argument(
    "-d", "--device",
    help="experiment on which device: cpu, cuda?",
  )
  args = parser.parse_args()
  return args

if __name__ == "__main__":

    #########################################################
    # Experiment argument parsing
    #########################################################
    args = getoptions()
    # global config
    config = {
        "input_dim" :   512,
        "num_batches":  100,
        "batch_size" :  8,
        "n_layers":     5,   
        "seq_len"   :  10,
    }

    #########################################################
    # MLP Experiment
    #########################################################
    # mlp = MLPEval(config["input_dim"], config["num_batches"], config["batch_size"], config["n_layers"], recompile=args.recompile)
    # mlp.eval()

    #########################################################
    # Convolution Experiment
    #########################################################
    conv = ConvEval(config["input_dim"], config["num_batches"], config["batch_size"])
    conv.eval()
    
    #########################################################
    # Transformer Experiment
    #########################################################
    # trans = TransformerEval(
    #   config["input_dim"], config["num_batches"], config["batch_size"], 
    #   config["input_dim"], config["seq_len"],
    #   recompile=args.recompile
    # )
    # trans.eval()