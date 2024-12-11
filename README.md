# Deep Learning System (10714) 
## Overview: `needle` ML framework

This directory contains our semester-long project for Deep Learning System (course# 10714). We have built a PyTorch-like machine learning framework called `needle` that supports: 
- `numpy`-like ndarray backend with `cpu` and `cuda` device support
- Tensor and tensor operators, similar to `torch.Tensor`
- Machine learning modules similar to `torch.nn.Module`. We support: 
  + Basic: `Linear`, `ReLU`, `BatchNorm`, `LayerNorm`, `Dropout`, `SoftmaxLoss`, `Residual`, `Sequential`.
  + Advanced: `Conv`, `MultiheadAttention`
- Data loader and dataset modules
- Reverse mode autodifferentiation (Reverse AD) and optimizer module (we support SGD and Adam)

Our final project extends `needle` with integration to the widely-adoped ML compilation framework Apache TVM. Here's what we extended:

```
python/
├── needle/
│   ├── ops/
│   │   └── ops_mathematics.py     
│   └── needle_tvm/
│       └── to_tvm.py             
apps/
├── tvm_eval.py                    
└── models/
    ├── mlp.py                     
    ├── resnet9.py                 
    ├── transformer.py             
    └── model_eval.py             
```

### `apps`
- [tvm_eval.py](./dlsys/apps/tvm_eval.py)  
    - Parses experiment arguments and settings for running evaluations
- [mlp.py](./dlsys/apps/models/mlp.py)
    - Defines the performance evaluation wrapper class for an MLP model
- [resnet9.py](./dlsys/apps/models/resnet9.py)
    - Defines the performance evaluation wrapper class for the ResNet9
- [transformer.py](./dlsys/apps/models/transformer.py)
    - Defines the performance evaluation wrapper class for a Transformer
- [model_eval.py](./dlsys/apps/models/model_eval.py)
    - Sets up model evaluation procedures and defines an auto-tuning method
### `Needle Backend`
- [ops_mathematics.py](./dlsys/python/needle/ops/ops_mathematic.py)
    - Extends the TensorOps class by adding the emit_te and emit methods
    - Constructs computation blocks in Relax IR using the BlockBuilder API
### `Needle TVM Extension`
- [to_tvm.py](./dlsys/python/needle_tvm/to_tvm.py)
    - Implements the to_tvm_tensor function to translate a computational graph into a Relax IR module
    - Performs topological tracing to ensure proper IR module construction

<!-- ### `backend_ndarray`
### `autograd` and `optim`
### `nn` and `init`
### `data` -->
### Example
#### Environment Setup
In the Google colab, mount your Google Drive and clone the Repository
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!mkdir -p 10714
%cd 10714
!git clone https://github.com/Theorem411/10714-project
%cd /content/drive/MyDrive/10714/10714-project/
!pip3 install pybind11
```
We recommend to install the package version of TVM from [mlc.ai](https://mlc.ai/chapter_tensor_program/case_study.html#install-packages)
```
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu122
python -c "from tvm import relax"
``` 
Set enviroment variables and Build the project
```
%set_env PYTHONPATH ./dlsys/python
%set_env NEEDLE_BACKEND nd

%cd /content/drive/MyDrive/10714/10714-project/dlsys
!make clean && make
```
#### Benchmark on CPU
**_Note:_** meta scheduling for `Transformer` and `ResNet9` model might take rounghly **_10-15 minutes_** to finish on the first run. However, since we reload the compiled module executable, the second time would be significantly faster as we bypass the meta scheduler.

If you want to recompile the model, add `-r` flag when running `tvm_eval.py`.

**MLP Performance**  
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='mlp' -d='cpu'
```
**Transformer Performance**
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='transformer' -d='cpu'
```
**ResNet9 Performance**
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='conv' -d='cpu'
```
#### Benchmark on GPU
To check if TVM have `USE_CUDA` turned on. You can run the following command and search for `USE_CUDA`.
```
!python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```
**MLP Performance**  
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='mlp' -d='cuda'
```
**Transformer Performance**
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='transformer' -d='cuda'
```
**ResNet9 Performance**
```
%cd /content/drive/MyDrive/10714/10714-project/dlsys/apps/
!python tvm_eval.py -m='conv' -d='cuda'
```

## Final Project: Integration with TVM
We explored two ways to integrate `needle` with TVM. Both ways uses the `BlockBuilder` API in `tvm.relax` to generate `IRModule` from `needle` models.

<details>
<summary> The translated IRModule: </summary>

```py
===== original module=====
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def te_broadcast_to(lv1: T.Buffer((T.int64(1), T.int64(512)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[T.int64(0), v_ax1])
                T.writes(T_broadcast_to[v_ax0, v_ax1])
                T_broadcast_to[v_ax0, v_ax1] = lv1[T.int64(0), v_ax1]

    @T.prim_func(private=True)
    def te_ewise_add(lv: T.Buffer((T.int64(32), T.int64(512)), "float32"), lv2: T.Buffer((T.int64(32), T.int64(512)), "float32"), T_add: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1], lv2[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv[v_ax0, v_ax1] + lv2[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def te_matmul(X: T.Buffer((T.int64(32), T.int64(512)), "float32"), B: T.Buffer((T.int64(512), T.int64(512)), "float32"), T_matmul: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, k in T.grid(T.int64(32), T.int64(512), T.int64(512)):
            with T.block("T_matmul"):
                v_ax0, v_ax1, v_k = T.axis.remap("SSR", [ax0, ax1, k])
                T.reads(X[v_ax0, v_k], B[v_k, v_ax1])
                T.writes(T_matmul[v_ax0, v_ax1])
                with T.init():
                    T_matmul[v_ax0, v_ax1] = T.float32(0.0)
                T_matmul[v_ax0, v_ax1] = T_matmul[v_ax0, v_ax1] + X[v_ax0, v_k] * B[v_k, v_ax1]

    @T.prim_func(private=True)
    def te_relu(lv3: T.Buffer((T.int64(32), T.int64(512)), "float32"), compute: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv3[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv3[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def te_reshape(A: T.Buffer((T.int64(1), T.int64(512)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
            with T.block("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[T.int64(0), v_ax1 % T.int64(512)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = A[T.int64(0), v_ax1 % T.int64(512)]

    @R.function
    def main(X: R.Tensor((32, 512), dtype="float32")) -> R.Tensor((32, 512), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.te_matmul, (X, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv1 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][1],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv2 = R.call_tir(cls.te_broadcast_to, (lv1,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv3 = R.call_tir(cls.te_ewise_add, (lv, lv2), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv4 = R.call_tir(cls.te_relu, (lv3,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv5 = R.call_tir(cls.te_matmul, (lv4, metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv6 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][3],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv7 = R.call_tir(cls.te_broadcast_to, (lv6,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv8 = R.call_tir(cls.te_ewise_add, (lv5, lv7), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv9 = R.call_tir(cls.te_relu, (lv8,), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            gv: R.Tensor((32, 512), dtype="float32") = lv9
            R.output(gv)
        return lv9
```

</details>

<details>
<summary> Translated IRModule after operator fusion: </summary>

```py
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu(X: T.Buffer((T.int64(32), T.int64(512)), "float32"), param_0: T.Buffer((T.int64(512), T.int64(512)), "float32"), lv1: T.Buffer((T.int64(1), T.int64(512)), "float32"), compute_intermediate: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_intermediate = T.alloc_buffer((T.int64(32), T.int64(512)))
        T_broadcast_to_intermediate = T.alloc_buffer((T.int64(32), T.int64(512)))
        T_add_intermediate = T.alloc_buffer((T.int64(32), T.int64(512)))
        for ax0, ax1, k in T.grid(T.int64(32), T.int64(512), T.int64(512)):
            with T.block("T_matmul"):
                v_ax0, v_ax1, v_k = T.axis.remap("SSR", [ax0, ax1, k])
                T.reads(X[v_ax0, v_k], param_0[v_k, v_ax1])
                T.writes(T_matmul_intermediate[v_ax0, v_ax1])
                with T.init():
                    T_matmul_intermediate[v_ax0, v_ax1] = T.float32(0.0)
                T_matmul_intermediate[v_ax0, v_ax1] = T_matmul_intermediate[v_ax0, v_ax1] + X[v_ax0, v_k] * param_0[v_k, v_ax1]
        for ax0, ax1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[T.int64(0), v_ax1])
                T.writes(T_broadcast_to_intermediate[v_ax0, v_ax1])
                T_broadcast_to_intermediate[v_ax0, v_ax1] = lv1[T.int64(0), v_ax1]
        for ax0, ax1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_intermediate[v_ax0, v_ax1], T_broadcast_to_intermediate[v_ax0, v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = T_matmul_intermediate[v_ax0, v_ax1] + T_broadcast_to_intermediate[v_ax0, v_ax1]
        for i0, i1 in T.grid(T.int64(32), T.int64(512)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(T_add_intermediate[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def te_reshape(A: T.Buffer((T.int64(1), T.int64(512)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(512)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
            with T.block("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[T.int64(0), v_ax1 % T.int64(512)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = A[T.int64(0), v_ax1 % T.int64(512)]

    @R.function
    def main(X: R.Tensor((32, 512), dtype="float32")) -> R.Tensor((32, 512), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][0],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv = R.call_tir(cls.fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu, (X, metadata["relax.expr.Constant"][1], lv1), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv6 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][2],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv1_1 = R.call_tir(cls.fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu, (lv, metadata["relax.expr.Constant"][3], lv6), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            R.output()
        return lv1_1
```

</details>

<details>
<summary> The translated IRModule after Meta-scheduling </summary>

```py
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu(X: T.Buffer((T.int64(32), T.int64(512)), "float32"), param_0: T.Buffer((T.int64(512), T.int64(512)), "float32"), lv1: T.Buffer((T.int64(1), T.int64(512)), "float32"), compute_intermediate: T.Buffer((T.int64(32), T.int64(512)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_intermediate = T.alloc_buffer((T.int64(32), T.int64(512)))
        for ax0_0_ax1_0_ax0_1_ax1_1_fused in T.parallel(T.int64(32)):
            for ax0_2_init, ax1_2_init, ax0_3_init in T.grid(T.int64(4), T.int64(8), T.int64(1)):
                for ax1_3_fused_init in T.vectorized(T.int64(16)):
                    with T.block("T_matmul_init"):
                        v_ax0 = T.axis.spatial(T.int64(32), ax0_0_ax1_0_ax0_1_ax1_1_fused // T.int64(8) * T.int64(8) + ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(4) // T.int64(2) * T.int64(4) + ax0_2_init + ax0_3_init)
                        v_ax1 = T.axis.spatial(T.int64(512), ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(8) // T.int64(4) * T.int64(256) + ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(2) * T.int64(128) + ax1_2_init * T.int64(16) + ax1_3_fused_init)
                        T.reads()
                        T.writes(T_matmul_intermediate[v_ax0, v_ax1])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        T_matmul_intermediate[v_ax0, v_ax1] = T.float32(0.0)
            for k_0, ax0_2, ax1_2, k_1, ax0_3 in T.grid(T.int64(8), T.int64(4), T.int64(8), T.int64(64), T.int64(1)):
                for ax1_3_fused in T.vectorized(T.int64(16)):
                    with T.block("T_matmul_update"):
                        v_ax0 = T.axis.spatial(T.int64(32), ax0_0_ax1_0_ax0_1_ax1_1_fused // T.int64(8) * T.int64(8) + ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(4) // T.int64(2) * T.int64(4) + ax0_2 + ax0_3)
                        v_ax1 = T.axis.spatial(T.int64(512), ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(8) // T.int64(4) * T.int64(256) + ax0_0_ax1_0_ax0_1_ax1_1_fused % T.int64(2) * T.int64(128) + ax1_2 * T.int64(16) + ax1_3_fused)
                        v_k = T.axis.reduce(T.int64(512), k_0 * T.int64(64) + k_1)
                        T.reads(T_matmul_intermediate[v_ax0, v_ax1], X[v_ax0, v_k], param_0[v_k, v_ax1])
                        T.writes(T_matmul_intermediate[v_ax0, v_ax1])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        T_matmul_intermediate[v_ax0, v_ax1] = T_matmul_intermediate[v_ax0, v_ax1] + X[v_ax0, v_k] * param_0[v_k, v_ax1]
        for i0_i1_fused_0 in T.parallel(T.int64(256)):
            for i0_i1_fused_1 in T.vectorized(T.int64(64)):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(32), (i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1) // T.int64(512))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1) % T.int64(512))
                    T.reads(T_matmul_intermediate[v_i0, v_i1], lv1[T.int64(0), v_i1])
                    T.writes(compute_intermediate[v_i0, v_i1])
                    compute_intermediate[v_i0, v_i1] = T.max(T_matmul_intermediate[v_i0, v_i1] + lv1[T.int64(0), v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def te_reshape(A: T.Buffer((T.int64(1), T.int64(512)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(512)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
            with T.block("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[T.int64(0), v_ax1 % T.int64(512)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = A[T.int64(0), v_ax1 % T.int64(512)]

    @R.function
    def main(X: R.Tensor((32, 512), dtype="float32")) -> R.Tensor((32, 512), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][0],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv = R.call_tir(cls.fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu, (X, metadata["relax.expr.Constant"][1], lv1), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            lv6 = R.call_tir(cls.te_reshape, (metadata["relax.expr.Constant"][2],), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv1_1 = R.call_tir(cls.fused_te_matmul_te_broadcast_to_te_ewise_add_te_relu, (lv, metadata["relax.expr.Constant"][3], lv6), out_sinfo=R.Tensor((32, 512), dtype="float32"))
            R.output()
        return lv1_1
```

</details>