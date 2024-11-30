import numpy as np
import tvm
from tvm import relax, transform, meta_schedule
from tvm import ir

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
from needle_tvm import *

import needle.nn as nn
from models import *

import time
from tqdm.auto import tqdm

device = ndl.cpu()

### MLP performance evaluation ######################
def evaluate_batch_mlp(model, module, X: np.ndarray):
    # input tensor wrapper
    input_ndl = ndl.Tensor(X, device=device, requires_grad=False, placeholder=True)
    input_tvm = tvm.nd.array(X)
    
    # performance
    start_time = time.perf_counter()
    ndl_out = model(input_ndl)
    ndl_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
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

### Transformer, RNN, LSTM performance evaluation ###
def evaluate_epoch_seq(model, module, dim, seq_len, num_batches, batch_size):
    """
      
    """
    np.random.seed(4)
    model.eval()

    ndl_time = 0
    tvm_time = 0

    for _ in range(num_batches):
        # set needle model to forward mode
        X = np.random.rand(seq_len, batch_size, dim).astype(np.float32)
        ndl_batch_time, tvm_batch_time = evaluate_batch_mlp(model, module, X)

        ndl_time += ndl_batch_time
        tvm_time += tvm_batch_time
    
    avg_ndl_time, avg_tvm_time = ndl_time / num_batches, tvm_time / num_batches
    print(f'\n{"-"*50} \nAVG NDL TIME: {avg_ndl_time} \tAVG TVM TIME: {avg_tvm_time} SPEEDUP: {avg_ndl_time / avg_tvm_time:.4f}\n\n')
    
    return 

### Meta schedule tuning ##############################
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
    
    # Return the optimized module
    return module 

def tune_tir_all(module, target, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp", max_funcs=5):
    # Iterate over all functions in the IRModule
    funcs = 0
    for func_name in tqdm(module.get_global_vars()):
        funcs += 1
        if max_funcs is not None and funcs > max_funcs: break
        try:
            func_name_str = func_name.name_hint
            print(f"tuning: {func_name_str}")
            # Create a tuning database for each function
            mod_func = tvm.IRModule.from_expr(module[func_name].with_attr("global_symbol", func_name_str))

            # Tune the TIR function
            database = meta_schedule.tune_tir(
                mod=mod_func,                 # Input module
                target=target,               # Target platform (e.g., "llvm", "cuda")
                max_trials_global=max_trials,  # Total tuning trials
                num_trials_per_iter=num_trials_per_iter,  # Trials per tuning iteration
                work_dir=f"{work_dir}/{func_name_str}",  # Separate logs for each function
            )

            # Compile the tuned TIR function into a new IRModule
            sch = meta_schedule.tir_integration.compile_tir(
                database=database,           # The tuning database
                mod=mod_func,                # Input module to compile
                target=target                # Target platform
            )

            # Update the module with the tuned function
            updated_mod = sch.mod["main"].with_attr("global_symbol", func_name_str)
            gv = module.get_global_var(func_name_str)
            module.update_func(gv, updated_mod)
        
        except:
            continue

    # Return the optimized module
    return module

### 
def evaluate_model(meval: ModelEval, args, device="cpu"):
  model = meval.model
  
  # choose device
  _ndl_device, _tvm_device = None, None
  if device == "cpu":
    _ndl_device = ndl.cpu()
    _tvm_device = tvm.target.Target("llvm")

  elif device == "cuda":
    _ndl_device = ndl.cuda()
    _tvm_device = tvm.target.Target("cuda -arch=sm_75")

  else: 
    raise NotImplementedError
  
  #########################################################
  # translate needle model to tvm IRModule
  #########################################################
  # dummy input
  x = meval.dummy_input()

  try: 
      if args.recompile:
        print(f"force recomile needle model to tvm!")
        assert False
      # try loading stored module as shared library
      module_ex = tvm.runtime.load_module(module_path)
      print(f"module reloaded from {module_path}")
  except: 
      # generate new tvm IRModule using Tensor graph
      module = to_tvm_tensor(model, True, ndl.Tensor(x, device=_ndl_device))
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
        tune_tir_all(module, "fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu", "llvm -num-cores=1", max_trials=5, num_trials_per_iter=5)
      print('='*5 + " auto-tuned module " + '='*5)
      module.show()

      # build and export the IRModule
      module_ex = relax.build(module, target=config["target"])
      module_ex.export_library(module_path)
      print(f"module exported to {module_path}")

  
