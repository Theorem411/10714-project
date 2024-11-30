import tvm
from tvm import relax
from tvm import te
from tvm.ir.transform import Sequential
from tvm.script import ir_module
from tvm.relax import op, Function, Var, block_builder

from collections import defaultdict
from collections import deque

import sys
sys.path.append('./python')

import needle as ndl
from needle.ops import *
from needle.nn import *

# symbolic tracer
import torch
from .tracer import NeedleTracer


def to_tvm_tensor(mod: Module, te: bool, *args, **kwargs):
  # IMPORTANT: to_tvm_tensor needs to mark explicitly which tensor is the input/placeholder
  for t in args: 
    if isinstance(t, Tensor):
      t.placeholder = True

  # run model to get output tensor
  output_tensor = mod(*args, **kwargs)

  # Topological sort on tensor graph: input first
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
    
  # initialize block builder to emit TensorIR function 
  bb = block_builder.BlockBuilder()
  fn_inputs = []
  fn_output = None
  value_to_var = {}

  # Create the "main" function in emitted IRModule 
  topo_order = topological_sort(output_tensor)
#   [print(node.op) for node in topo_order]
#   print(f"topo_order: {topo_order}")
#   [print(f"leaf or op : {x.is_leaf() or x.op} is placeholder?: {x.placeholder} \n {x} \n{'-'*30}\n") for x in topo_order]
  with bb.function("main"):
      with bb.dataflow():
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
              # print(f'op: {repr(node.op)}\n')
              # tvm_var = node.op.emit_te(bb, value_to_var, node)
              print(f"value to var: {value_to_var}")
              print(node.op)
              if te:
                tvm_var = node.op.emit_te(bb, value_to_var, node)
                print(tvm_var)
              else:
                tvm_var = node.op.emit(bb, value_to_var, node)
              if tvm_var is not None:
                 print(f"node: {node} tvm_var: {tvm_var}")
              else:
                 print(f"node: {node} tvm_var: None")
              value_to_var[node] = tvm_var
      
          fn_output = bb.emit_output(value_to_var[topo_order[-1]])
      print(f"value_to_var: {value_to_var}")
      bb.emit_func_output(value_to_var[topo_order[-1]], fn_inputs)
  return bb.get()

def to_tvm_fx(mod: Module, *args, **kwargs):
  # code adopted from: https://mlc.ai/chapter_integration/index.html#remark-translating-into-high-level-operators 
  def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, relax.TensorStructInfo(shape, "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()

  graph = NeedleTracer.trace(mod)
  traced = torch.fx.GraphModule(mod, graph)
  raise NotImplementedError

