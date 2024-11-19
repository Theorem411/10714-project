# `Tracer` and `symbolic_trace`
This documentation records what we learned from `torch.fx.Tracer` and `torch.fx.symbolic_trace`. 

# `Tracer.trace` workflow
1.	Initialize a Tracer: An instance of the Tracer class is created.
2.	Trace the Module/Function: The Tracer’s trace method is called with the module or function as an argument.
3.	Create Graph Nodes: As the function or module is executed, the Tracer intercepts operations and creates nodes in the graph.
4.	Return a GraphModule: A GraphModule is created using the traced graph and returned.

## setup in the `trace` method
Steps in the trace Method

1.	Setup:
+	Determines whether the root is an nn.Module or a function.
+	Stores the root object.
+	Initializes an empty Graph.
2.	Prepare for Tracing:
+	Collects tensor attributes from the root module (if applicable).
+	Unwraps the function to get to the innermost callable (handles decorators).
+	Analyzes the function signature to prepare arguments.
3.	Create Placeholders:
+	Calls create_args_for_root to create placeholder nodes corresponding to the function’s arguments.
+	Handles *args and **kwargs if present.
4.	Set Up Patching:
+	Patches certain methods to intercept calls to modules and functions.
+	Overrides torch.nn.Module.__call__ and torch.nn.Module.__getattr__ to track module calls and attribute accesses.
5.	Execute the Function:
+	Calls the function with the prepared arguments.
+	As the function executes, operations are intercepted, and corresponding nodes are added to the graph.
6.	Finalize the Graph:
+	Adds an output node to the graph.
+	Cleans up patches and returns the constructed graph.

## How Operations Are Intercepted

+	`Proxy` Objects: The Tracer uses Proxy objects to wrap tensors and track operations performed on them.
+ Method Overriding: By overriding `__call__` and `__getattr__`, the `Tracer` intercepts calls to submodules and accesses to parameters/buffers.

## Inspecting and Parsing the forward Function

When tracing an nn.Module, the Tracer needs to inspect and parse the forward method. Here’s how it does it:

Unwrapping the Function:
+ The trace method uses `inspect.unwrap` to remove any decorators and get to the original forward method.

```[python]
import inspect 
# remove any decorators and get to the original forward method.
fn_for_analysis = inspect.unwrap(root_fn)

# get the parameters of the function. Determines the number of arguments and their names.
sig = inspect.signature(fn_for_analysis)
```

## Creating Placeholder Nodes
+ For each parameter in the function signature, a placeholder node is created in the graph.
```[python]
def proxy_placeholder(name):
    return self._proxy_placeholder(name, concrete_args, sig, fn_for_analysis)
```

## Call function with proxy arguments
```[python]
self.create_node(
    "output",
    "output",
    (self.create_arg(fn(*args)),),
    {},
    type_expr=fn.__annotations__.get("return", None),
)
```

## Executing the Function with Proxies:
```[python]
self.create_node(
    "output",
    "output",
    (self.create_arg(fn(*args)),),
    {},
    type_expr=fn.__annotations__.get("return", None),
)
```

## Patching Methods for Interception

To intercept operations inside the forward method, the Tracer patches certain methods:

**Patching `torch.nn.Module.__call__`**:
+ Overrides the `__call__` method to intercept calls to submodules.

**Patching `torch.nn.Module.__getattr__`**:
Overrides the `__getattr__` method to intercept accesses to parameters and buffers. Uses `getattr` to return a proxy for parameters and buffers.
```[python]
def module_getattr_wrapper(mod, attr):
    attr_val = _orig_module_getattr(mod, attr)
    return self.getattr(attr, attr_val, parameter_proxy_cache)
```

**Patching Global Functions:**
Uses a patcher to wrap functions so that when they’re called with a `Proxy`, the call is recorded in the graph.
```[python]
def _patch_wrapped_functions(patcher):
    # ...
    patcher.patch(frame_dict, name, _create_wrapped_func(value))
```

# How Python Inspects the Function Body

Python provides several introspection tools that torch.fx leverages:

Using `inspect` Module

+	`inspect.getsource(fn)`: Retrieves the source code of the function `fn`.
+	`inspect.getclosurevars(fn)`: Gets the closure variables used in `fn`.
+	`inspect.getmembers(object)`: Gets the members of an `object`.

However, `torch.fx` mainly relies on executing the function with proxies rather than parsing the source code.

Limitations:
+ `torch.fx` does not parse the abstract syntax tree (AST) of the function.
+ It works by executing the function with proxies that record the operations.

# Handling Control Flow

+ Since `torch.fx` operates by executing the function, it can only capture the control flow that is actually executed.
+ If there are conditional statements or loops, only the paths taken during execution are traced.
+ To handle different control flows, you may need to use concrete_args or other techniques.