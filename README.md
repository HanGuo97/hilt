# CuTeDSL Utilities

A Python package for CuTeDSL utilities.

## Installation

```bash
git clone --recurse-submodules https://github.com/HanGuo97/cutedsl_utilities
cd cutedsl_utilities
pip install -e .
```

## Example

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
