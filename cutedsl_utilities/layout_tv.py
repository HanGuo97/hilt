import matplotlib.pyplot as plt
from cutedsl_utilities.pycute_utils import (
    Layout,
    idx2crd,
    crd2idx,
    product,
    visualize_layout,
    default_color_map,
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
    use_index: bool = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:

    assert len(layout_tv.shape) == 2
    assert len(layout_tv.stride) == 2

    def get_crd_and_idx(index: int) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
        thr_crd, val_crd = idx2crd(
            idx=index,
            shape=layout_tv.shape,
            stride=layout_tv.stride,
        )
        thr_idx = crd2idx(
            crd=thr_crd,
            shape=layout_tv.shape[0],
        )
        val_idx = crd2idx(
            crd=val_crd,
            shape=layout_tv.shape[1],
        )
        return thr_crd, val_crd, thr_idx, val_idx

    def color_map(index: int) -> tuple[float, float, float, float]:
        thr_crd, val_crd, thr_idx, val_idx = get_crd_and_idx(index)
        rgb = default_color_map(thr_idx)
        alpha = (val_idx + 1) / product(layout_tv.shape[1])
        return *rgb, alpha

    def label_map(index: int) -> str:
        thr_crd, val_crd, thr_idx, val_idx = get_crd_and_idx(index)
        if not use_index:
            return f"T: {thr_crd}\nV: {val_crd}"
        else:
            return f"T: {thr_idx}\nV: {val_idx}"

    tiler_layout = Layout(tiler_mn)
    return visualize_layout(
        layout=tiler_layout,
        color_map=color_map,
        label_map=label_map,
        **kwargs
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
