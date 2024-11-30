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
        "batch_size" :  1,  # 8,
        "dim" :         4, # 8,
        "n_layers":     2,   # 2
        "activation":   nn.ReLU,
        "seq_len"   :  10,
        "bias":         True,
        "device" :      ndl.cpu(),
        "target" :      tvm.target.Target("llvm"),
        "tvm_device":   tvm.cpu(),
        "num_batches":  100
    }

    # input
    x = np.random.randint(low=0, high=config["seq_len"], size=(config["seq_len"], config["batch_size"], config["dim"])).astype(np.float32)

    #########################################################
    # Needle model
    #########################################################
    model = Transformer(embedding_size=config["dim"], hidden_size=config["dim"], num_layers=config["n_layers"], sequence_len = config["seq_len"], device=config["device"])


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

    # # compile IRModule
    # with transform.PassContext(opt_level=4):
    #   print('='*5 + " Apply meta_schedule..." + '='*5)
    #   tune_tir(module, "fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu", "llvm -num-cores=1", max_trials=5, num_trials_per_iter=5)
    # print('='*5 + " auto-tuned module " + '='*5)
    # module.show()

    # build and execute the IRModule
    module_ex = relax.build(module, target=config["target"])
    module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
    
    # evaluate average runtime across batches
    X_out = evaluate_epoch_mlp(model, module_vm, dim=config["dim"], num_batches=config["num_batches"], batch_size=config["batch_size"])