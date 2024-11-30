"""hw1/apps/simple_ml.py"""

import struct
import gzip
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
from utils import *

import time
device = ndl.cpu()

if __name__ == "__main__":
    #########################################################
    # Performance Benchmarking
    #########################################################
    config = {
        "batch_size" :  32,  # 8,
        "dim" :         512, # 8,
        "n_layers":     8,   # 2
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
    model = MLPModel(dim=config["dim"], device=config["device"], bias=config["bias"])


    #########################################################
    # Needle model
    #########################################################
    # generate tvm IRModule using Tensor graph
    module = to_tvm_tensor(model, True, ndl.Tensor(x, device=config["device"]))
    print('='*5 + " original module" + '='*5)
    module.show()

    # optimize IRModule
    # module = tune_tir(module, "te_matmul", target=config["target"])
    # module.show()

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

    # compile IRModule
    with transform.PassContext(opt_level=4):
      print('='*5 + " Apply meta_schedule..." + '='*5)
      # Tune the specified TIR function
      # database = meta_schedule.tune_tir(
      #     mod=module,                 
      #     target=config["target"],    
      #     max_trials_global=64,  # Total tuning trials
      #     num_trials_per_iter=64,  # Trials per tuning iteration
      #     work_dir="./mlp-meta-sched",          # Directory to store logs
      # )

      # module = meta_schedule.tir_integration.compile_tir(database, module, target=config["target"])
      tune_tir(module, "fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu", "llvm -num-cores=1", max_trials=5, num_trials_per_iter=5)
    print('='*5 + " auto-tuned module " + '='*5)
    module.show()

    # build and execute the IRModule
    module_ex = relax.build(module, target=config["target"])
    module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
    
    # evaluate average runtime across batches
    X_out = evaluate_epoch_mlp(model, module_vm, dim=config["dim"], num_batches=config["num_batches"], batch_size=config["batch_size"])