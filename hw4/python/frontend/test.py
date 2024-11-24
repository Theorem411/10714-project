import tvm
from tvm import relax
from tvm import te
from tvm.script import ir_module
from tvm.relax import op
import sys
import numpy as np
from collections import defaultdict

sys.path.append('./python')

import needle as ndl
from needle.ops import *
from needle.nn import *

def convert_tensor_to_tvm(tensor):
    numpy_array = tensor.numpy() 
    return tvm.nd.array(numpy_array)

def map_op_to_tvm(op):
    op_mapping = {
        "add": add,
        "matmul": matmul,
        "ReLU": relu,
        "Reshape": reshape
        # we'd add any other ops here any ops from ops.mathematic or ops.logarithmic
    }
    if op.__class__.__name__ not in op_mapping:
        raise ValueError(f"Unsupported operation: {op.__class__.__name__}")
    return op_mapping[op.__class__.__name__]

from tvm.relax import Function, Var, block_builder
from collections import deque

def convert_graph_to_tvm(output_tensor):
    bb = block_builder.BlockBuilder()

    # Map of `Value` nodes to Relax variables
    value_to_var = {}

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

    # Process the graph in topological order
    topo_order = topological_sort(output_tensor)
    [print(f"is leaf? : {x.is_leaf()} is placeholder?: {x.placeholder} \n {x} \n{'-'*30}\n") for x in topo_order]
    for node in topo_order:
        # Leaf nodes (inputs or constants)
        if node.is_leaf():
            if node.placeholder:
                tvm_var = (relax.Var("X", relax.TensorStructInfo(node.shape, "float32")))
                value_to_var.get(node, tvm_var)
                continue
            else:
                tvm_var = (relax.const(node.numpy(), relax.TensorStructInfo(node.shape, "float32")))
                value_to_var.get(node, tvm_var)
                continue

        # Map the operation to TVM
        print(f'op: {repr(node.op)}')
        tvm_op = map_op_to_tvm(node.__class__)

        # Get TVM inputs by recursively converting dependencies

        tvm_inputs = [value_to_var[inp] for inp in node.inputs]

        # Emit the Relax operation
        tvm_var = bb.emit(tvm_op(*tvm_inputs))
        value_to_var[node] = tvm_var

    # Create the Relax function
    with bb.function("main"):
        bb.emit_output(value_to_var[output_tensor])
    return bb.get()


class MyModel(Module):
    def __init__(self, dim, device=ndl.cpu()):
        self.fc = Linear(dim, dim, device=device, bias=False)
        self.act = ReLU()
    
    def forward(self, x):
        return self.act(self.fc(x))

if __name__ == "__main__":
    model = MyModel(5)
    input_tensor = Tensor(np.array([[1, 2, 3, 4, 5]]), device=ndl.cpu(), requires_grad=False, placeholder=True)
    print(f'input: {input_tensor.shape}')
    output_tensor = model(input_tensor)

    # Convert the output tensor to TVM
    tvm_func = convert_graph_to_tvm(output_tensor)

    # Compile the TVM function
    tvm_mod = tvm.relay.build(tvm_func, target="llvm")

    # Run the TVM function
    tvm_output = tvm_mod(input_tensor.numpy())
    print(tvm_output)
    print(output_tensor.numpy())
    assert tvm_output == output_tensor.numpy()
 



