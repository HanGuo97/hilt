import matplotlib.pyplot as plt
from collections import defaultdict
from cutedsl_utilities.pycute_utils import (
    Layout,
    idx2crd,
    crd2idx,
    product,
    visualize_layout,
)

TilerMN = tuple[int, int]
TVShape = tuple[tuple[int, int] | int, tuple[int, int] | int]
LayoutTV = tuple[TVShape, TVShape]

__all__ = [
    "visualize_layout_tv",
    "visualize_layout_tv_maybe_duplicates",
    "tiler_crd_to_layout_tv_crd",
]


def visualize_layout_tv(
    tiler_mn: TilerMN,
    layout_tv: Layout,
    use_alpha: bool = False,
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
        _, _, thr_idx, val_idx = get_crd_and_idx(index)
        colors = plt.cm.Set2.colors
        rgb = colors[thr_idx % len(colors)]
        if use_alpha:
            vals = product(layout_tv.shape[1])
            alpha = (vals - val_idx) / vals
        else:
            alpha = 1.0
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


def visualize_layout_tv_maybe_duplicates(
    tiler_mn: TilerMN,
    layout_tv: Layout,
    use_alpha: bool = False,
    use_index: bool = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:

    assert len(layout_tv.shape) == 2
    assert len(layout_tv.stride) == 2
    thr_shape, val_shape = layout_tv.shape
    inverse_tv = defaultdict(list)
    for thr_idx in range(product(thr_shape)):
        thr_crd = idx2crd(
            idx=thr_idx,
            shape=thr_shape,
        )
        for val_idx in range(product(val_shape)):
            val_crd = idx2crd(
                idx=val_idx,
                shape=val_shape,
            )
            index = layout_tv(thr_idx, val_idx)
            inverse_tv[index].append((thr_crd, val_crd, thr_idx, val_idx))

    def color_map(index: int) -> tuple[float, float, float, float]:
        # using the first entry to decide color
        _, _, thr_idx, val_idx = inverse_tv[index][0]
        colors = plt.cm.Set2.colors
        rgb = colors[thr_idx % len(colors)]
        if use_alpha:
            vals = product(layout_tv.shape[1])
            alpha = (vals - val_idx) / vals
        else:
            alpha = 1.0
        return *rgb, alpha

    def label_map(index: int) -> str:
        thr_crds = []
        val_crds = []
        thr_idxs = []
        val_idxs = []
        for thr_crd, val_crd, thr_idx, val_idx in inverse_tv[index]:
            thr_crds.append(str(thr_crd))
            val_crds.append(str(val_crd))
            thr_idxs.append(str(thr_idx))
            val_idxs.append(str(val_idx))

        if not use_index:
            return f"T: {' | '.join(thr_crds)}\nV: {' | '.join(val_crds)}"
        else:
            return f"T: {' | '.join(thr_idxs)}\nV: {' | '.join(val_idxs)}"

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
