import tvm
from tvm import relax
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))


import needle as ndl
import needle.nn as nn
from needle_tvm import to_tvm_tensor, to_tvm_fx

from contextlib import contextmanager
import time

@contextmanager
def timer():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


# !global config. edit this to edit run hyperparams
config = {
    "batch_size" : 8,
    "dim" : 8,
    "n_layers": 2,
    "activation": nn.ReLU,
    "device" : ndl.cpu,
}


class MLPModel(nn.Module):
    def __init__(self, n_layers=3, dim=512, activation=nn.ReLU, bias=False, device=None):
            components = []
            for _ in range(n_layers):
                components.extend([nn.Linear(dim, dim, bias=bias, device=device), activation()])
            self.mlp = nn.Sequential(*components)
    def forward(self, x):
        return self.mlp(x)

if __name__ == "__main__":
    #########################################################
    # Performance Benchmarking
    #########################################################
    # !important set placeholder = True
    x = np.random.rand(config["batch_size"], config["dim"]).astype(np.float32)
    input_tensor = ndl.Tensor(x, device=config["device"](), requires_grad=False, placeholder=True)
    # print(f'input: {input_tensor.shape}')

    mod = MLPModel(dim=config["dim"], device=config["device"]())
    output_tensor = mod(input_tensor)

    # generate tvm IRModule using Tensor graph
    MLPModule = to_tvm_tensor(mod, input_tensor)
    MLPModule.show()

    # Compile the TVM IRModule
    ex = relax.build(MLPModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # Run the TVM IRModule
    tvm_output = vm["main"](tvm.nd.array(x))
    assert np.allclose(tvm_output.asnumpy(),output_tensor.numpy())
    print(f"Outputs Match")

    # ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
    # print("MyModelWithParams_before time-cost: %g ms" % (ftimer(tvm.nd.array(x)).mean * 1000))