import numpy as np
import tvm
from tvm import relax, transform, meta_schedule
from tvm import ir

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import needle as ndl
import needle.nn as nn

import time
from tqdm.auto import tqdm

class ModelEval:
  model: nn.Module
  module: tvm.IRModule

  ndl_device: ndl.ndarray.BackendDevice
  tvm_device: tvm.device
  tvm_target: tvm.Target

  input_dim: int
  num_batches: int
  batch_size: int

  def __init__(self, input_dim, num_batches, batch_size, 
               device="cpu", recompile=False): 
    # detect how many cores in current environment
    self.cores = os.cpu_count()
    
    # necessary params
    self.input_dim = input_dim
    self.num_batches = num_batches
    self.batch_size = batch_size

    # choose device
    if device == "cpu":
      self.ndl_device = ndl.cpu()
      self.tvm_device = tvm.cpu()
      self.tvm_target = tvm.target.Target("llvm")
    elif device == "cuda":
      self.ndl_device = ndl.cuda()
      self.tvm_device = tvm.cuda()
      self.tvm_target = tvm.target.Target("llvm")
    else: 
      raise NotImplementedError

    # construct model
    self.model = self.construct_model()

    # translate model to tvm irmodule and save as shared library
    self.module_path = self.module_save_path()
    
    try: 
      if not recompile:
        module_ex = tvm.runtime.load_module(self.module_path)
        self.module = relax.VirtualMachine(module_ex, self.tvm_device)
        print(f"module reloaded from {self.module_path}")
      else:
        self.module = compile_model()
        print(f"module recompiled")
    except FileNotFoundError: 
      self.module = compile_model()
      print(f"module compiled")

  @property
  def model():
    return model

  @property
  def module():
    return module

  # must be overriden by children class
  def construct_model(self): 
    raise NotImplementedError

  def module_save_path(self):
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the current script
    module_lib_path = os.path.join(current_file_path, "module_lib")
    os.makedirs(module_lib_path, exist_ok=True)
    return os.path.join(module_lib_path, self.module_lib_save_name())
  # must be overriden by children class
  def module_lib_save_name(self):
    raise NotImplementedError

  # can be override for different model input data type
  def dummy_input(self):
    raise NotImplementedError

  ### compile TVM irmodule functionality #####################################
  def compile_model(self):
    # generate new tvm IRModule using Tensor graph
    ir_module = to_tvm_tensor(model, True, ndl.Tensor(x, device=self.ndl_device))
    print('='*5 + " original module" + '='*5)
    ir_module.show()
    
    # optimize module: peephole optimization, operator fusion
    ir_module = self.opt_irmodule(ir_module)
    print('='*5 + " transformed module" + '='*5)
    ir_module.show()

    # meta-scheduling
    with transform.PassContext(opt_level=4):
      print('='*5 + " Apply meta_schedule..." + '='*5)
      self.tune_tir_all(ir_module)
      print('='*5 + " auto-tuned module " + '='*5)
      ir_module.show()
    
    # build and export module as library
    module_ex = relax.build(ir_module, target=self.tvm_target)
    module_ex.export_library(self.module_path)
    print(f"module exported to {self.module_path}")

    self.module = relax.VirtualMachine(module_ex, self.tvm_device)
    
  # children class should override this function to provide model-specific optimizations
  def opt_irmodule(self, ir_module: tvm.IRModule):
    pipeline = self._default_pipeline()
    return pipeline(ir_module)

  def _default_pipeline():
    return tvm.ir.transform.Sequential([
      tvm.relax.transform.LegalizeOps(),
      tvm.relax.transform.AnnotateTIROpPattern(),
      tvm.relax.transform.FuseOps(),
      tvm.relax.transform.FuseTIR(),
    ])


  ### metaschedule tuning                  #####################################
  def tune_tir_all(self, ir_module, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp", max_funcs=None):
    # add number of cores
    target = self.target.with_attr("num-cores", self.cores)
    print(f"{__func__}: target={target}")
    
    # Iterate over all functions in the IRModule
    funcs = 0
    for func_name in tqdm(ir_module.get_global_vars()):
        funcs += 1
        if max_funcs is not None and funcs > max_funcs: break
        try:
            func_name_str = func_name.name_hint
            print(f"tuning: {func_name_str}")
            # Create a tuning database for each function
            mod_func = tvm.IRModule.from_expr(ir_module[func_name].with_attr("global_symbol", func_name_str))

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
            gv = ir_module.get_global_var(func_name_str)
            ir_module.update_func(gv, updated_mod)
        
        except:
            continue
  

  ### performance evaluation              #####################################
  def eval_batch(self, X: np.ndarray):
    # input tensor wrapper
    input_ndl = ndl.Tensor(X, device=self.ndl_device, requires_grad=False, placeholder=True)
    input_tvm = tvm.nd.array(X)
    
    # performance
    start_time = time.perf_counter()
    ndl_out = self.model(input_ndl)
    ndl_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    tvm_out = self.module["main"](input_tvm)
    tvm_time = time.perf_counter() - start_time
    
    # correctness test: 
    try: 
      assert np.allclose(tvm_out.asnumpy(),ndl_out.numpy(), atol=1e-4) # tweak tolerance if fails
    except AssertionError: 
      # Compute the absolute difference between two outputs
      abs_diff = np.abs(np.linalg.norm(tvm_out.asnumpy()) - np.linalg.norm(ndl_out.numpy()))
      print(f"TVM-NDL diff norm: {abs_diff}")
      raise ValueError

    return ndl_time, tvm_time

  def eval(self):
    # !NECESSARY: fix all rand seed to pass correctness after module reload
    np.random.seed(4)

    model.eval()

    # init timing code
    ndl_time = 0
    tvm_time = 0

    for _ in range(self.num_batches):
        # set needle model to forward mode
        X = self.dummy_input()
        ndl_batch_time, tvm_batch_time = self.eval_batch(X)

        ndl_time += ndl_batch_time
        tvm_time += tvm_batch_time
    
    avg_ndl_time, avg_tvm_time = ndl_time / self.num_batches, tvm_time / self.num_batches
    print(f'\n\n\n {"-"*50} \nAVG NDL TIME: {avg_ndl_time} \tAVG TVM TIME: {avg_tvm_time}')
