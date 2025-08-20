import matplotlib.pyplot as plt
from cutedsl_utilities.pycute_utils import (
    Layout,
    idx2crd,
    crd2idx,
    visualize_layout,
)

TilerMN = tuple[int, int]
TVShape = tuple[tuple[int, int] | int, tuple[int, int] | int]
LayoutTV = tuple[TVShape, TVShape]

__all__ = [
    "visualize_layout_tv",
    "tiler_crd_to_layout_tv_crd",
]


def visualize_layout_tv(
    tiler_mn: TilerMN,
    layout_tv: Layout,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:

    def label_map(index: int) -> str:
        thr_crd, val_crd = idx2crd(
            idx=index,
            shape=layout_tv.shape,
            stride=layout_tv.stride,
        )
        return f"T: {thr_crd}\nV: {val_crd}"

    tiler_layout = Layout(tiler_mn)
    return visualize_layout(
        layout=tiler_layout,
        label_map=label_map,
    )


def tiler_crd_to_layout_tv_crd(
    tiler_crd: TilerMN,
    tiler_mn: TilerMN,
    layout_tv_shape: TVShape,
    layout_tv_stride: TVShape,
) -> TVShape:
    assert len(tiler_crd) == 2
    assert len(tiler_mn) == 2
    assert len(layout_tv_shape) == 2
    assert len(layout_tv_stride) == 2
    tiler_idx = crd2idx(
        crd=tiler_crd,
        shape=tiler_mn,
    )
    return idx2crd(
        idx=tiler_idx,
        shape=layout_tv_shape,
        stride=layout_tv_stride,
    )
