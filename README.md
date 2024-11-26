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

Our final project extends `needle` with integration to the widely-adoped ML compilation framework Apache TVM.

### `backend_ndarray`
### `autograd` and `optim`
### `nn` and `init`
### `data`
### Example

## Final Project: Integration with TVM
We explored two ways to integrate `needle` with TVM. Both ways uses the `BlockBuilder` API in `tvm.relax` to generate `IRModule` from `needle` models.

