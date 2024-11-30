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

from contextlib import contextmanager
import time

import torch

# Timer
@contextmanager
def timer(model_name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()


class ConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, linear_output_dim=10, device=None):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride, padding, device=device)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * ((32 - kernel_size + 2 * padding) // stride + 1) ** 2, linear_output_dim, device=device)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Performance evaluation
def evaluate_batch_conv(model, module, X: np.ndarray):
    input_ndl = ndl.Tensor(X, device=ndl.cpu(), requires_grad=False, placeholder=True)
    input_tvm = tvm.nd.array(X)

    start_time = time.perf_counter()
    with timer("needle"):
        ndl_out = model(input_ndl)
    ndl_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    with timer("tvm"):
        tvm_out = module["main"](input_tvm)
    tvm_time = time.perf_counter() - start_time

    assert np.allclose(tvm_out.asnumpy(), ndl_out.numpy(), atol=1e-4)
    return ndl_time, tvm_time

def evaluate_epoch_conv(model, module, input_shape, num_batches):
    model.eval()

    ndl_time = 0
    tvm_time = 0

    for _ in range(num_batches):
        X = np.random.rand(*input_shape).astype(np.float32)
        ndl_batch_time, tvm_batch_time = evaluate_batch_conv(model, module, X)

        ndl_time += ndl_batch_time
        tvm_time += tvm_batch_time

    avg_ndl_time = ndl_time / num_batches
    avg_tvm_time = tvm_time / num_batches
    print(f"\n\n\n{'-'*50}\nAVG NDL TIME: {avg_ndl_time}\tAVG TVM TIME: {avg_tvm_time}")
    return

# Compile and tune TIR
def tune_tir(module, func_name, target, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp"):
    mod_func = tvm.IRModule.from_expr(module[func_name].with_attr("global_symbol", "main"))

    database = meta_schedule.tune_tir(
        mod=mod_func,
        target=target,
        max_trials_global=max_trials,
        num_trials_per_iter=num_trials_per_iter,
        work_dir=work_dir,
    )

    sch = meta_schedule.tir_integration.compile_tir(
        database=database,
        mod=mod_func,
        target=target
    )

    updated_mod = sch.mod["main"].with_attr("global_symbol", func_name)
    gv = module.get_global_var(func_name)
    module.update_func(gv, updated_mod)
    return module

if __name__ == "__main__":
    #########################################################
    # Performance Benchmarking
    #########################################################
    config = {
        "batch_size": 32,
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "input_height": 32,
        "input_width": 32,
        "device": ndl.cpu(),
        "target": tvm.target.Target("llvm"),
        "tvm_device": tvm.cpu(),
        "num_batches": 100,
    }

    # Input tensor shape
    input_shape = (
        config["batch_size"],
        config["in_channels"],
        config["input_height"],
        config["input_width"],
    )

    #########################################################
    # Needle model
    #########################################################
    model = ConvModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
        device=config["device"],
    )

    torch_model = torch.nn.Conv2d(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
        bias=False,
    )
    torch_model.eval()

    #########################################################
    # TVM Module
    #########################################################
    weight = np.random.rand(config["out_channels"], config["in_channels"], config["kernel_size"], config["kernel_size"]).astype(np.float32)
    torch_model.weight.data = torch.tensor(weight)

    x = np.random.rand(*input_shape).astype(np.float32)
    torch_input = torch.tensor(x)
    tvm_input = tvm.nd.array(x)

    module = to_tvm_tensor(model, False, ndl.Tensor(x, device=config["device"]))
    print("=" * 5 + " Original module " + "=" * 5)
    module.show()

    # # Optimize module
    # module = tvm.relax.transform.LegalizeOps()(module)
    # module = tvm.ir.transform.Sequential(
    #     [
    #         tvm.relax.transform.AnnotateTIROpPattern(),
    #         tvm.relax.transform.FuseOps(),
    #         tvm.relax.transform.FuseTIR(),
    #     ]
    # )(module)
    # print("=" * 5 + " Transformed module " + "=" * 5)
    # module.show()

    # Tune TIR
    # tune_tir(
    #     module,
    #     "fused_te_conv2d_te_relu",  # Adjust function name if needed
    #     "llvm -num-cores=1",
    #     max_trials=5,
    #     num_trials_per_iter=5,
    # )
    # print("=" * 5 + " Auto-tuned module " + "=" * 5)
    # module.show()

    # Build and execute
    module_ex = relax.build(module, target=config["target"])
    module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])

    torch_output = torch_model(torch_input).detach().numpy()
    tvm_output = module_vm["main"](tvm_input).asnumpy()

    print(np.allclose(tvm_output, torch_output, atol=1e-4))

    # # Evaluate performance
    # evaluate_epoch_conv(model, module_vm, input_shape=input_shape, num_batches=config["num_batches"])
