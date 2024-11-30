"""hw1/apps/simple_ml.py"""
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
from needle_tvm import *

import needle.nn as nn
from models import *
from utils import *

np.random.seed(0)

def getoptions():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-r", "--recompile",
    action="store_true",
    help="recompile tvm module",
  )
  args = parser.parse_args()
  return args

if __name__ == "__main__":
    #########################################################
    # Experiment argument parsing
    #########################################################
    args = getoptions()

    
    # input
    x = np.random.rand(config["batch_size"], config["dim"]).astype(np.float32)

    #########################################################
    # Needle model
    #########################################################
    model = MLPModel(n_layers=config["n_layers"], dim=config["dim"], device=config["device"], bias=config["bias"])

    #########################################################
    # TVM model
    #########################################################
    # IRModule load/store path
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the current script
    current_file_path = os.path.join(current_file_path, "module_lib")
    os.makedirs(current_file_path, exist_ok=True)
    module_path = os.path.join(current_file_path, f"mlp-{config['device'].name}.so")
    
    try: 
      if args.recompile:
        assert False
      # try loading stored module as shared library
      module_ex = tvm.runtime.load_module(module_path)
      print(f"module reloaded from {module_path}")
    except: 
      # generate new tvm IRModule using Tensor graph
      module = to_tvm_tensor(model, True, ndl.Tensor(x, device=config["device"]))
      print('='*5 + " original module" + '='*5)
      module.show()

      # optimize IRModule
      module = tvm.relax.transform.LegalizeOps()(module)
      module = tvm.ir.transform.Sequential(
        [
          tvm.relax.transform.AnnotateTIROpPattern(),
          tvm.relax.transform.FuseOps(),
          tvm.relax.transform.FuseTIR(),
        ])(module)
      print('='*5 + " transformed module" + '='*5)

      module.show()

      # try metaschedule on fused operator
      with transform.PassContext(opt_level=4):
        print('='*5 + " Apply meta_schedule..." + '='*5)

        # module = meta_schedule.tir_integration.compile_tir(database, module, target=config["target"])
        tune_tir(module, "fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu", "llvm -num-cores=1", max_trials=5, num_trials_per_iter=5)
      print('='*5 + " auto-tuned module " + '='*5)
      module.show()
    
      # build and export the IRModule
      module_ex = relax.build(module, target=config["target"])
      module_ex.export_library(module_path)
      print(f"module exported to {module_path}")

    #########################################################
    # Compare Needle and TVM model perforamnce
    #########################################################
    module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
    # evaluate average runtime across batches
    X_out = evaluate_epoch_mlp(model, module_vm, dim=config["dim"], num_batches=config["num_batches"], batch_size=config["batch_size"])