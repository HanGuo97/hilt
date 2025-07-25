# CuTeDSL Utilities

A Python package for CuTeDSL utilities.

## Installation

```bash
git clone --recurse-submodules https://github.com/HanGuo97/cutedsl-utilities
cd cutedsl_utilities
pip install -e .
```

## Example - Visualize TV Layout

```python
import torch
import cutedsl_utilities
latex, source = cutedsl_utilities.visualize_layout_tv(
    tiler_mn=(32, 32),
    layout_tv=(
        ((16,   8), (2,  4)),
        ((2 , 128), (1, 32)),
    ),
    dtype=torch.float16,
)
```
![Example](images/layout-tv-example.png)


# Profile-Kernel

A CLI tool to profile CUDA kernels using NVIDIA Nsight Compute, extract source-level performance information, and automatically highlight performance bottlenecks (e.g. excessive memory access or warp stalls).

## Usage - Profile Kernel

```bash
profile-kernel \
  --ncu_path /path/to/ncu \
  --exe_path /path/to/python \
  --filepath my_kernel.py \
  --output_filename output \
  [--filter access|stall|warning]  # optional
```

## Example Output - Profile Kernel

The output CSV will contain columns like:

```
#, Warning Type, Warning Info, Address, SASS, ...
```