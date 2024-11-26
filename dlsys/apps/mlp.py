import tvm
from tvm import relax
import numpy as np

import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
from needle_tvm import to_tvm_tensor, to_tvm_fx

class MLPModel(nn.Module):
    def __init__(self, n_layers=3, dim=512, activation=nn.ReLU, bias=False, device=None):
        self.mlp = nn.Sequential(*[(nn.Linear(dim, dim, bias=bias), activation()) for _ in range(n_layers)])
    
    def forward(self, x):
        return self.mlp(x)

if __name__ == "__main__":
    #########################################################
    # Performance Benchmarking
    #########################################################

    # !important set placeholder = True
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    input_tensor = Tensor(x, device=ndl.cpu(), requires_grad=False, placeholder=True)
    # print(f'input: {input_tensor.shape}')

    mod = MLPModel(5, device=ndl.cpu())

    # generate tvm IRModule using Tensor graph
    MLPModule = to_tvm_tensor(mod, input_tensor)
    MLPModule.show()

    # Compile the TVM IRModule
    ex = relax.build(MLPModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # Run the TVM IRModule
    tvm_output = vm["main"](tvm.nd.array(x))
    print(tvm_output)
    print(output_tensor.numpy())
    assert np.allclose(tvm_output.asnumpy(),output_tensor.numpy())
 