# `Proxy`
This documentation records what we learned from the design of `torch.fx.proxy`. 

**key components:**
1.	`Scope` and `ScopeContextManager`
2.	`TracerBase` Class
3.	`GraphAppendingTracer` Class
4.	`TraceError` Exception
5.	`Proxy` Class
6.	`Attribute` Class
7.	`ParameterProxy` Class
8.	Magic Methods for `Proxy`


# `Scope` and `ScopeContextManager`

## `Scope` Class

The `Scope` class represents the context in which a Node in the computational graph is created. It keeps track of: 
+ `module_path`: A string representing the hierarchical path to the current module (e.g., "`module.submodule`").
+ `module_type`: The class type of the current module.

This information is essential for understanding where in the module hierarchy a particular operation originates, which is useful for debugging and optimization.

```[python]
class Scope:
    def __init__(self, module_path: str, module_type: Any):
        self.module_path = module_path
        self.module_type = module_type
```

## `ScopeContextManager` Class

`ScopeContextManager` is a context manager that updates the current Scope when entering or exiting a module during tracing. It ensures that the `Scope` reflects the correct module hierarchy at any point in time.

```[python]
class ScopeContextManager:
    def __init__(self, scope: Scope, current_scope: Scope):
        self._prev_scope = copy.copy(scope)
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        self._scope = scope

    def __enter__(self):
        return self._scope

    def __exit__(self, *args):
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
```

# `TracerBase` Class

The `TracerBase` class is the foundation of the symbolic tracing mechanism. It defines how operations are intercepted and recorded into the Graph. Subclasses can override its methods to customize tracing behavior.

## Key Attributes
+ `graph`: An instance of Graph where nodes are recorded.
+ `record_stack_traces`: If True, stack traces are recorded for each node.
+ `check_mutable_operations`: If True, mutable operations are checked during tracing.
+ `trace_asserts`: If True, assertions are traced.
+ `proxy_buffer_attributes`: If True, accesses to buffer attributes are proxied.
+ `traced_func_name`: The name of the function to trace (default is "forward").
+ `scope`: An instance of Scope representing the current scope.
+ `module_stack`: An ordered dictionary representing the module call stack.
+ `node_name_to_scope`: A mapping from node names to their scope information.

## Important Methods
### `create_node`
Creates a Node in the Graph. This method can be overridden to add custom behavior, such as validation or additional metadata.

```[python]
def create_node(self, kind: str, target: Target, args: Tuple[Argument, ...],
                kwargs: Dict[str, Argument], name: Optional[str] = None,
                type_expr: Optional[Any] = None) -> Node:
    # Custom behavior can be added here
    node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
    self.node_name_to_scope[node.name] = (
        self.scope.module_path,
        self.scope.module_type,
    )
    # Record stack traces if enabled
    if self.record_stack_traces and not node.stack_trace:
        node.stack_trace = ''.join(CapturedTraceback.extract().format())
    return node
```

### `proxy`
Wraps a `Node` into a `Proxy` object. This allows intercepted operations to be recorded into the graph while appearing transparent to the user code.

```[python]
def proxy(self, node: Node) -> 'Proxy':
    return Proxy(node, self)
```

### `create_proxy`
The core method that creates a proxy for an operation. It handles the creation of the node and wraps it in a `Proxy` object.

```[python]
def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                 name: Optional[str] = None, type_expr: Optional[Any] = None,
                 proxy_factory_fn: Callable[[Node], 'Proxy'] = None):
    args_ = self.create_arg(args)
    kwargs_ = self.create_arg(kwargs)
    node = self.create_node(kind, target, args_, kwargs_, name, type_expr)
    if not proxy_factory_fn:
        proxy = self.proxy(node)
    else:
        proxy = proxy_factory_fn(node)
    # Record stack traces if enabled
    if self.record_stack_traces and not proxy.node.stack_trace:
        proxy.node.stack_trace = ''.join(CapturedTraceback.extract().format())
    return proxy
```

### `create_arg`
Converts Python objects used as arguments into `Argument` types suitable for the IR. It handles recursive structures and special types.

```[python]
def create_arg(self, a: Any) -> Argument:
    if not isinstance(a, Proxy) and hasattr(a, '__fx_create_arg__'):
        return a.__fx_create_arg__(self)
    elif isinstance(a, tuple) and hasattr(a, '_fields'):
        args = tuple(self.create_arg(elem) for elem in a)
        return type(a)(*args)
    elif isinstance(a, (tuple, list)):
        return type(a)(self.create_arg(elem) for elem in a)
    elif isinstance(a, dict):
        return {self.create_arg(k): self.create_arg(v) for k, v in a.items()}
    elif isinstance(a, slice):
        return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))
    elif isinstance(a, range):
        return range(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))
    elif isinstance(a, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        return a
    elif isinstance(a, Proxy):
        return a.node
    elif is_dataclass(a):
        kwargs = {field.name: self.create_arg(getattr(a, field.name)) for field in fields(a)}
        return self.create_node("call_function", a.__class__, (), kwargs)
    elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
        return a
    raise NotImplementedError(f"argument of type: {type(a)}")
```

### `to_bool`
Handles conversion of a proxy to a boolean, which occurs in control flow contexts (e.g., if statements). By default, raises a `TraceError`.
```[python]
def to_bool(self, obj: 'Proxy') -> bool:
    raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
```

### `iter`
Handles iteration over a proxy object. By default, raises a `TraceError` since the length and contents are unknown during tracing.

```[python]
def iter(self, obj: 'Proxy') -> Iterator:
    raise TraceError('Proxy object cannot be iterated. This can be '
                     'attempted when the Proxy is used in a loop or'
                     ' as a *args or **kwargs function argument.')
```

### `keys`
Handles calls to the `keys()` method on a proxy, used when unpacking dictionaries (`**kwargs`).
```[python]
def keys(self, obj: 'Proxy') -> Any:
    return Attribute(obj, 'keys')()
```


# `GraphAppendingTracer` class
A subclass of `TracerBase`, the `GraphAppendingTracer` is used when you already have a `Graph` object and want to append nodes to it without performing a full trace.

```
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}
```

# `TraceError` Exception
An exception class used to signal errors during tracing, especially when an operation cannot be symbolically traced (e.g., trying to iterate over a `Proxy`).

```[python]
class TraceError(ValueError):
    pass
```

# `Proxy` Class
The `Proxy` class is central to symbolic tracing. It wraps `Node` objects and overloads various operators and methods to intercept operations and record them into the computational graph.

**Key Concepts:**
+	`Operation Recording`: When operations are performed on a `Proxy`, they are intercepted, and corresponding nodes are added to the graph.
+	`Transparency`: Proxies aim to behave like the underlying objects, so user code doesn’t need to be modified extensively.

## Important Methods and Attributes
### `__init__`: 
Initializes the `Proxy` with a `Node` and an optional `Tracer`.
```[python]
def __init__(self, node: Node, tracer: 'Optional[TracerBase]' = None):
    if tracer is None:
        tracer = GraphAppendingTracer(node.graph)
    self.tracer = tracer
    self.node = node
```

### `__getattr__`:
Intercepts attribute access on the `proxy`. Returns an Attribute `proxy` to handle the attribute access.

```[python]
def __getattr__(self, k) -> 'Attribute':
    return Attribute(self, k)
```

### `__call__`
Intercepts function calls on the proxy.
```[python]
def __call__(self, *args, **kwargs) -> 'Proxy':
    return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)
```

### `__torch_function__`
```[python]
@classmethod
def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    tracers: Dict[Any, None] = {}

    def find_tracer(a):
        if isinstance(a, cls):
            tracers[a.tracer] = None
    torch.fx.node.map_aggregate(args, find_tracer)
    torch.fx.node.map_aggregate(kwargs, find_tracer)

    if len(tracers) > 1:
        raise RuntimeError('Found multiple different tracers')
    tracer = next(iter(tracers.keys()))

    if isinstance(orig_method, torch._C.ScriptMethod):
        args = (orig_method.owner,) + args
        return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
    if torch.overrides.is_tensor_method_or_property(orig_method):
        return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
    else:
        if isinstance(orig_method, torch._ops.HigherOrderOperator):
            raise RuntimeError("Unable to symbolically trace HigherOrderOperators")
        return tracer.create_proxy('call_function', orig_method, args, kwargs, name=tracer.graph._target_to_str(orig_method.__name__))
```

# `Attribute` Class
The `Attribute` class is a subclass of Proxy used specifically for attribute accesses on a proxy object. It handles both attribute retrieval and method calls.

# `ParameterProxy` Class
The `ParameterProxy` class is a specialized proxy for `torch.nn.Parameter` objects. It allows certain attribute accesses to pass through to the underlying parameter, enabling conditional tests on attributes like `shape` or `size`.

**Key Features**:
+ Allows access to attributes like `shape`, `size`, `dim`, `ndim`, `numel`, and `nelement`.
+	Useful for tracing code that depends on parameter attributes.

```[python]
class ParameterProxy(Proxy):
    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert isinstance(param, torch.nn.Parameter)
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f'ParameterProxy({self.name})'

    @property
    def shape(self):
        return self.param.shape

    def size(self):
        return self.param.size()

    def dim(self):
        return self.param.dim()

    @property
    def ndim(self):
        return self.param.ndim

    def numel(self):
        return self.param.numel()

    def nelement(self):
        return self.param.nelement()
```

# Magic Methods for Proxy
To ensure that operations on proxies are correctly recorded, magic methods (like `__add__`, `__mul__`, etc.) are dynamically added to the Proxy class. These methods intercept operations and create corresponding nodes in the graph.

## Adding Magic Methods
### Direct Magic Methods 
For each method in magic_methods, a corresponding method is added to Proxy.
```[python]
for method in magic_methods:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl)
    _scope(method)
```
### Reflectable Magic Methods
For operations where the proxy is on the right-hand side (e.g., other + proxy), reflectable magic methods are added.

```[python]
def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)

for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
```

### Example: 
If `__add__` is in magic_methods, then the following method is added to `Proxy`:
```[python]
def __add__(self, other):
    return self.tracer.create_proxy('call_function', operator.add, (self, other), {})
```