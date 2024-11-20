# `Node`, `Graph`, and `GraphModule`

This documentation records what we learned from the design of `torch.fx.Graph`, `torch.fx.Node`. This should shed light on how we designed our `needle.Graph` and `needle.Node` module.

## How everything fits together

A `Graph` is a data structure that represents a method on a `GraphModule`. The information that this requires is:

+ inputs to the method

+ operations that run inside the method

+ output (i.e. return) value from the method

All three of these concepts (inputs, operations, and outputs) are represented with `Node` instances.

## Graph Module

`GraphModule` is an `nn.Module` generated from an `fx.Graph`. `Graphmodule` has a `graph` attribute, as well as `code` and `forward` attributes generated from that graph.

### **Attributes**

+ **`graph`**:
  + The symbolic representation of the computation.
  + Allows inspection and modification of the computational graph.
+ **`forward()`**:
  + The auto-generated forward method based on the `Graph`.
+ **`code`**:
  + Human-readable Python code for the forward method, generated from the graph.

### **Key Features**

1. **Execution**:
   + A `GraphModule` can be executed like any PyTorch module using `forward()`.
2. **Graph Introspection**:
   + You can inspect the underlying graph using `graph` or `code` attributes.
3. **Dynamic Modification**:
   + Modify the graph to add optimizations, change operations, or apply custom transformations.

## Graph

+ Input: FX, method inputs are specified via special `placeholder` nodes
+ Intermediate operations: The `get_attr`, `call_function`, `call_module`, and `call_method` nodes represent the operations in the method. More information on these will be provided in the node section.
+ Output: The return value in a Graph is specified by a special `output` node.

## Node

The `Node` class represents a single operation in a `Graph`. Depending on the operation type (`op`), it can represent inputs, outputs, or various function calls. Below is an explanation of the key node types and how they differ.

---

### **Node Types**

#### **1. `placeholder`**

+ Represents an input to the function or model.
+ **Example**: A parameter like `x` in a function definition `def forward(x, y):`.
+ **Key Attributes**:
  + `name`: The name assigned to this input (e.g., `"x"`).
  + `target`: The argument name (e.g., `"x"`).
  + `args`: Contains the default value (if any) for this input.
  + `kwargs`: Not used for `placeholder`.

---

#### **2. `get_attr`**

+ Used to fetch attributes (e.g., parameters, buffers) from the module hierarchy.
+ **Example**: Accessing a parameter like `self.weight` in a module.
+ **Key Attributes**:
  + `name`: The name of the variable assigned to the fetched attribute.
  + `target`: The fully-qualified name of the attribute in the module hierarchy (e.g., `"linear.weight"`).
  + `args`, `kwargs`: Not used for `get_attr`.

---

#### **3. `call_function`**
+ Represents a call to a **free function** (standalone function not tied to a specific object or module).
+ **Example**: Calling `torch.relu(x)`.
+ **Key Attributes**:
  + `name`: The name assigned to the result of the function call.
  + `target`: The function being called (e.g., `torch.relu`).
  + `args`: The positional arguments passed to the function.
  + `kwargs`: The keyword arguments passed to the function.
+ **When to Use**: For calls to global functions like `torch.add`, `torch.sigmoid`, or `math.sqrt`.

---

#### **4. `call_module`**
+ Represents a call to an **`nn.Module`** within the module hierarchy.
+ **Example**: Applying `self.linear(x)` in a module, where `self.linear` is an instance of `nn.Linear`.
+ **Key Attributes**:
  + `name`: The name assigned to the result of the module call.
  + `target`: The fully-qualified name of the module in the hierarchy (e.g., `"linear"`).
  + `args`: The arguments passed to the module’s `forward()` method, excluding `self`.
  + `kwargs`: The keyword arguments passed to the module’s `forward()` method.
+ **When to Use**: For layers or submodules defined in the `nn.Module` hierarchy, such as `nn.Linear`, `nn.Conv2d`, or custom modules.

---

#### **5. `call_method`**
+ Represents a call to a **method** on a specific object or tensor.
+ **Example**: Calling `x.view(1, -1)` or `x.matmul(y)` on tensors.
+ **Key Attributes**:
  + `name`: The name assigned to the result of the method call.
  + `target`: The name of the method as a string (e.g., `"view"`, `"matmul"`).
  + `args`: The positional arguments passed to the method, including `self` as the first argument.
  + `kwargs`: The keyword arguments passed to the method.
+ **When to Use**: For operations that involve methods on objects, typically on tensors or user-defined classes.

---

#### **6. `output`**
+ Represents the return value of the function or model.
+ **Example**: The `return` statement in a function or the output tensor of a model.
+ **Key Attributes**:
  + `args[0]`: The returned value.
  + `target` and `kwargs`: Not used for `output`.

---

### **Key Differences Between Call Nodes**

| Node Type      | What It Represents                         | Example Code              | Key Difference                                           |
|----------------|--------------------------------------------|---------------------------|---------------------------------------------------------|
| `call_function`| Calls a global, free-standing function     | `torch.relu(x)`           | The function is not tied to a specific object or module.|
| `call_module`  | Invokes an `nn.Module`’s `forward()` method| `self.linear(x)`          | Refers to a submodule in the module hierarchy.          |
| `call_method`  | Invokes a method on an object              | `x.view(1, -1)`           | Calls a method tied to the object or tensor (`x`).      |

---
