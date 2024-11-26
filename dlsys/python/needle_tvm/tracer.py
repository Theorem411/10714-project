import torch 
from torch.fx import symbolic_trace, GraphModule, Tracer
from torch.fx.graph import Node, Graph


class NeedleTracer(Tracer):
  pass