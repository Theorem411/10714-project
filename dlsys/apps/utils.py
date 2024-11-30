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

### MLP performance evaluation ###
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
    
    # Return the optimized module
    return module 

def tune_tir_all(module, target, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp"):
    # Iterate over all functions in the IRModule
    for func_name in tqdm(module.get_global_vars()):
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

    # Return the optimized module
    return module