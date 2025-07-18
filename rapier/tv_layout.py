from typing import Tuple
from ._C import visualize_layout_tv as _visualize_layout_tv

TilerMN = Tuple[int, int]
LayoutTV = Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]


def visualize_layout_tv(
    tiler_mn: TilerMN,
    layout_tv: LayoutTV,
) -> str:
    return _visualize_layout_tv(tiler_mn, layout_tv)
