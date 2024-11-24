import tvm
from tvm import relax
from tvm import te
from tvm.ir.transform import Sequential
from tvm.script import ir_module
from tvm.relax import op
import sys
import numpy as np
from collections import defaultdict

sys.path.append('./python')

import needle as ndl
from needle.ops import *
from needle.nn import *

from tvm.relax import Function, Var, block_builder
from collections import deque

def convert_tensor_to_tvm(tensor):
    numpy_array = tensor.numpy() 
    return tvm.nd.array(numpy_array)

def convert_graph_to_tvm(output_tensor):
    # Topological sort of the graph
    def topological_sort(output):
        visited = set()
        topo_order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for inp in node.inputs:
                dfs(inp)
            topo_order.append(node)

        dfs(output)
        return topo_order  # Reverse for topological order
    
    # tvm block builder for IRModule 
    bb = block_builder.BlockBuilder()
    fn_inputs = []
    fn_output = None

    # Create the Relax function "main"
    with bb.function("main"):
        with bb.dataflow():
            # Map of `Value` nodes to Relax variables
            value_to_var = {}


            # Process the graph in topological order
            topo_order = topological_sort(output_tensor)
            [print(f"leaf or op : {x.is_leaf() or x.op} is placeholder?: {x.placeholder} \n {x} \n{'-'*30}\n") for x in topo_order]
            for i, node in enumerate(topo_order):
                # Leaf nodes (inputs or constants)
                if node.is_leaf():
                    if node.placeholder:
                        tvm_var = (relax.Var("X", relax.TensorStructInfo(node.shape, "float32")))
                        value_to_var.setdefault(node, tvm_var)
                        fn_inputs.append(tvm_var)
                        continue
                    else:
                        tvm_var = (relax.const(node.numpy(), relax.TensorStructInfo(node.shape, "float32")))
                        value_to_var.setdefault(node, tvm_var)
                        continue

                # Map the operation to TVM
                print(f'op: {repr(node.op)}\n')
                tvm_var = node.op.map_tvm(bb, value_to_var, node)

                # tvm_op = map_op_to_tvm(node.__class__)
                # # Get TVM inputs by recursively converting dependencies
                # tvm_inputs = [value_to_var[inp] for inp in node.inputs]

                # Emit the Relax operation
                # tvm_var = bb.emit(tvm_op(*tvm_inputs))
                value_to_var[node] = tvm_var
        
            fn_output = bb.emit_output(value_to_var[topo_order[-1]])
        bb.emit_func_output(value_to_var[topo_order[-1]], fn_inputs)
    return bb.get()


class MyModel(Module):
    def __init__(self, dim, device=ndl.cpu()):
        self.backbone = Sequential(
            Linear(5, 5, bias=False),
            ReLU(),
            Linear(5,5, bias=False),
            ReLU()
        )
    
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = MyModel(5)
    # !important! set placeholder = True
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    input_tensor = Tensor(x, device=ndl.cpu(), requires_grad=False, placeholder=True)
    print(f'input: {input_tensor.shape}')
    output_tensor = model(input_tensor)

    # Convert the output tensor to TVM
    tvm_func = convert_graph_to_tvm(output_tensor)
    tvm_func.show()

    # Compile the TVM function
    ex = relax.build(tvm_func, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # Run the TVM function
    tvm_output = vm["main"](tvm.nd.array(x))
    print(tvm_output)
    print(output_tensor.numpy())
    assert np.allclose(tvm_output.asnumpy(),output_tensor.numpy())
 



