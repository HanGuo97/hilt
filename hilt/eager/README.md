```diff
import torch
- import cutlass.cute as cute
+ import rapier.eager.api as cute
from rapier.eager.tensor import from_torch
```

```python
layout0 = cute.make_layout(shape=(2, 3), stride=(1, 3))
layout1 = cute.make_layout(shape=(3, 2), stride=(2, 1))
layout2 = cute.blocked_product(layout0, layout1)
layout3 = cute.composition(layout0, layout1)
layout4 = cute.right_inverse(layout3)

print(layout0)  # (2,3):(1,3)
print(layout1)  # (3,2):(2,1)
print(layout2)  # ((2,3),(3,2)):((1,18),(3,9))
print(layout3)  # (3,2):(3,1)
print(layout4)  # 2:3

layout0.visualize(dpi=200)
layout1.visualize(dpi=200)
layout2.visualize(dpi=200)
layout3.visualize(dpi=200)
layout4.visualize(dpi=200)
```

```python
tensor = torch.randn(7, 15, dtype=torch.bfloat16)
cute_tensor = from_torch(tensor, memspace="gmem")
cute_fragment = cute.make_fragment_like(cute_tensor, cute.Float16)
cute_ssatensor = cute_fragment.load()

print(f"{cute_tensor!r}")  # tensor<ptr<bf16, gmem> o (7,15):(15,1)>
print(f"{cute_fragment!r}")  # tensor<ptr<f16, rmem> o (7,15):(1,7)>
print(f"{cute_ssatensor!r}")  # tensor_value<vector<105xf16> o (7, 15)>
```