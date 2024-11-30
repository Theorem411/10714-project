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

from contextlib import contextmanager
import time

## Timer 
@contextmanager
def timer(model_name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    # print(f"{model_name} Execution time: {end_time - start_time:.6f} seconds")


device = ndl.cpu()

### MLP performance evaluation ###
def evaluate_batch_mlp(model, module, X: np.ndarray):
    # input tensor wrapper
    input_ndl = ndl.Tensor(X, device=device, requires_grad=False, placeholder=True)
    input_tvm = tvm.nd.array(X)
    
    # performance
    start_time = time.perf_counter()
    with timer("needle"):
      ndl_out = model(input_ndl)
    ndl_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    with timer("tvm"):
      tvm_out = module["main"](input_tvm)
    tvm_time = time.perf_counter() - start_time
    # correctness: 
    assert np.allclose(tvm_out.asnumpy(),ndl_out.numpy(), atol=1e-4) # tweak tolerance if fails

    return ndl_time, tvm_time

    
    
def evaluate_epoch_mlp(model, module, dim, num_batches, batch_size):
    """
      
    """
    np.random.seed(4)
    model.eval()

    ndl_time = 0
    tvm_time = 0

    for _ in range(num_batches):
        # set needle model to forward mode
        X = np.random.rand(batch_size, dim).astype(np.float32)
        ndl_batch_time, tvm_batch_time = evaluate_batch_mlp(model, module, X)

        ndl_time += ndl_batch_time
        tvm_time += tvm_batch_time
    
    avg_ndl_time, avg_tvm_time = ndl_time / num_batches, tvm_time / num_batches
    print(f'\n\n\n {"-"*50} \nAVG NDL TIME: {avg_ndl_time} \tAVG TVM TIME: {avg_tvm_time}')
    
    return 

def tune_tir(module, func_name, target, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp"):
    # Create a tuning database
    mod_func = tvm.IRModule.from_expr(module[func_name].with_attr("global_symbol", "main"))

    # Tune the specified TIR function
    database = meta_schedule.tune_tir(
        mod=mod_func,                 # Input module
        target=target,              # Target platform (e.g., "llvm", "cuda")
        max_trials_global=max_trials,  # Total tuning trials
        num_trials_per_iter=num_trials_per_iter,  # Trials per tuning iteration
        work_dir=work_dir,          # Directory to store logs
    )

    # Compile the tuned TIR function into a new IRModule
    sch = meta_schedule.tir_integration.compile_tir(
        database=database,          # The tuning database
        mod=mod_func,                 # Input module to compile
        target=target               # Target platform
    )

    updated_mod = sch.mod["main"].with_attr("global_symbol", func_name)
    gv = module.get_global_var(func_name)
    module.update_func(gv, updated_mod)


def argparse():
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
    args = argparse()

    #########################################################
    # Performance Benchmarking
    #########################################################
    config = {
        "batch_size" :  32,  # 8,
        "dim" :         512, # 8,
        "n_layers":     2,
        "activation":   nn.ReLU,
        "bias":         True,
        "device" :      ndl.cpu(),
        "target" :      tvm.target.Target("llvm"),
        "tvm_device":   tvm.cpu(),
        "num_batches":  100
    }

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
    module_lib_dir = os.makedirs("./module_lib", exist_ok=True)
    module_path = os.path.join(module_lib_dir, f"mlp-{config["device"]}.so")
    
    try: 
      # try loading stored module as shared library
      module_ex = tvm.runtime.load_module(module_path)
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

    #########################################################
    # Compare Needle and TVM model perforamnce
    #########################################################
    module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
    # evaluate average runtime across batches
    X_out = evaluate_epoch_mlp(model, module_vm, dim=config["dim"], num_batches=config["num_batches"], batch_size=config["batch_size"])