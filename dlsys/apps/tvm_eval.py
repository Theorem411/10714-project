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
    type=str,
    default="cpu",
    help="experiment on which device: cpu, cuda?",
  )
  parser.add_argument(
    "-m", "--model",
    type=str,
    default="mlp",
    help="Specify the model to evaluate. Default is 'mlp'."
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
        "in_channels": 3,
        "out_channels": 16,
        "input_height": 32,
        "input_width": 32,
    }

    if args.model == "mlp": 
      # MLP Experiment
      mlp = MLPEval(config["input_dim"], config["num_batches"], config["batch_size"], config["n_layers"],
                    device=args.device, recompile=args.recompile)
      mlp.evaluate_performance()
    elif args.model == "conv":
      # Convolution Experiment
      conv = ConvEval(config["input_dim"], config["num_batches"], config["batch_size"], 
                    config["in_channels"], config["out_channels"], config["input_height"], config["input_width"],
                    device=args.device, recompile=args.recompile)
      conv.evaluate_performance()
    elif args.model == "transformer":
      # Transformer Experiment
      trans = TransformerEval(config["input_dim"], config["num_batches"], config["batch_size"], 
                    config["input_dim"], config["seq_len"],
                    device=args.device, recompile=args.recompile)
      trans.evaluate_performance()
    else: 
      raise NotImplementedError