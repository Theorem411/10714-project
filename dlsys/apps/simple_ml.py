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

from contextlib import contextmanager
import time
import tvm.meta_schedule as ms
from tvm.meta_schedule.database import MemoryDatabase

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

### PTB training ###
# def get_batch(batches, i, bptt, device=None, dtype=None):
#     """
#     get_batch subdivides the source data into chunks of length bptt.
#     If source is equal to the example output of the batchify function, with
#     a bptt-limit of 2, we'd get the following two Variables for i = 0:
#     ┌ a g m s ┐ ┌ b h n t ┐
#     └ b h n t ┘ └ c i o u ┘
#     Note that despite the name of the function, the subdivison of data is not
#     done along the batch dimension (i.e. dimension 1), since that was handled
#     by the batchify function. The chunks are along dimension 0, corresponding
#     to the seq_len dimension in the LSTM or RNN.
#     Inputs:
#     batches - numpy array returned from batchify function
#     i - index
#     bptt - Sequence length
#     Returns:
#     data - Tensor of shape (bptt, bs) with cached data as NDArray
#     target - Tensor of shape (bptt*bs,) with cached data as NDArray
#     """
#     ### BEGIN YOUR SOLUTION
#     seq_len = min(bptt, len(batches) - 1 - i)
#     data = batches[i:i+seq_len, :]
#     target = batches[i+1:i+1+seq_len, :].flatten()
#     return ndl.Tensor(data, device=device, dtype=dtype), ndl.Tensor(target, device=device, dtype=dtype)

# def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
#         clip=None, device=None, dtype="float32"):
#     """
#     Iterates over the data. If optimizer is not None, sets the
#     model to train mode, and for each batch updates the model parameters.
#     If optimizer is None, sets the model to eval mode, and simply computes
#     the loss/accuracy.

#     Args:
#         data: data of shape (nbatch, batch_size) given from batchify function
#         model: LanguageModel instance
#         seq_len: i.e. bptt, sequence length
#         loss_fn: nn.Module instance
#         opt: Optimizer instance (optional)
#         clip: max norm of gradients (optional)

#     Returns:
#         avg_acc: average accuracy over dataset
#         avg_loss: average loss over dataset
#     """
#     np.random.seed(4)
#     ### BEGIN YOUR SOLUTION
#     nbatch, batch_size = data.shape
#     losses = []
#     correct = 0
#     total = 0

#     for i in range(nbatch):
#         if opt is not None:
#             model.train()
#             opt.reset_grad()
#         else:
#             model.eval()
        
#         X, y = get_batch(data, i, seq_len, device=device, dtype=dtype)
#         if (X.shape[0] < 1): break
#         out, _ = model.forward(X)
#         loss = loss_fn(out, y)

#         if opt is not None:
#             loss.backward()
#             opt.step()

#         losses.append(loss.numpy().item())
#         correct += np.sum(out.numpy().argmax(axis=1) == y)
#         total += y.shape[0]
    
#     avg_loss = np.mean(losses)
#     avg_acc = correct / total
#     return avg_acc, avg_loss
    
#     ### END YOUR SOLUTION


# def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
#           lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
#           device=None, dtype="float32"):
#     """
#     Performs {n_epochs} epochs of training.

#     Args:
#         model: LanguageModel instance
#         data: data of shape (nbatch, batch_size) given from batchify function
#         seq_len: i.e. bptt, sequence length
#         n_epochs: number of epochs (int)
#         optimizer: Optimizer class
#         lr: learning rate (float)
#         weight_decay: weight decay (float)
#         loss_fn: nn.Module class
#         clip: max norm of gradients (optional)

#     Returns:
#         avg_acc: average accuracy over dataset from last epoch of training
#         avg_loss: average loss over dataset from last epoch of training
#     """
#     for _ in range(n_epochs):
#         avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), optimizer(model.parameters(), lr=lr, weight_decay=weight_decay),
#             clip, device, dtype)
#     return avg_acc, avg_loss
#     ### BEGIN YOUR SOLUTION
    
#     ### END YOUR SOLUTION

# def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
#         device=None, dtype="float32"):
#     """
#     Computes the test accuracy and loss of the model.

#     Args:
#         model: LanguageModel instance
#         data: data of shape (nbatch, batch_size) given from batchify function
#         seq_len: i.e. bptt, sequence length
#         loss_fn: nn.Module class

#     Returns:
#         avg_acc: average accuracy over dataset
#         avg_loss: average loss over dataset
#     """
#     np.random.seed(4)
#     ### BEGIN YOUR SOLUTION
#     return epoch_general_ptb(data, model, seq_len, loss_fn())
#     ### END YOUR SOLUTION
def tune_tir(module, target, max_trials=64, num_trials_per_iter=64, work_dir="./tune_tmp"):
    # Create a tuning database
    database = MemoryDatabase()

    # Iterate through all functions in the IRModule
    for gv in module.get_global_vars():
        func_name = gv.name_hint
        func = module[gv]

        if not isinstance(func, tvm.tir.PrimFunc):
            # Skip non-TIR functions
            continue

        # Create a new IRModule containing just this function
        mod_func = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))

        # Tune the TIR function
        database = ms.tune_tir(
            mod=mod_func,                 # Input module
            target=target,               # Target platform
            max_trials_global=max_trials,  # Total tuning trials
            num_trials_per_iter=num_trials_per_iter,  # Trials per tuning iteration
            work_dir=work_dir,           # Directory to store logs
        )

        # Compile the tuned TIR function into a new IRModule
        sch = ms.tir_integration.compile_tir(
            database=database,          # The tuning database
            mod=mod_func,               # Input module to compile
            target=target               # Target platform
        )

        # Update the original function in the module with the tuned version
        updated_func = sch.mod["main"].with_attr("global_symbol", func_name)
        module.update_func(gv, updated_func)

    # Return the optimized module
    return module

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
    module = to_tvm_tensor(model, False, ndl.Tensor(x, device=config["device"]))
    print('='*5 + " original module" + '='*5)
    module.show()

    # optimize IRModule
    module = tune_tir(module, target=config["target"])
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

    # compile IRModule
    # with transform.PassContext(opt_level=4):
    #   print('='*5 + " Apply meta_schedule..." + '='*5)
    #   database = meta_schedule.Database.create_workload(module)
    #   module = meta_schedule.apply(module, target=target, database=database)
    # module_ex = relax.build(module, target=config["target"])
    # module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
    with transform.PassContext(opt_level=4):
      module_ex = relax.build(module, target=config["target"])
      print('='*5 + " transformed module" + '='*5)
      # module_ex.ir_mod.show()
      module_vm = relax.VirtualMachine(module_ex, config["tvm_device"])
      # prof.run(lambda: module_vm["main"](tvm.nd.array(x)))
      # prof.show()

    
    # evaluate average runtime across batches
    X_out = evaluate_epoch_mlp(model, module_vm, dim=config["dim"], num_batches=config["num_batches"], batch_size=config["batch_size"])