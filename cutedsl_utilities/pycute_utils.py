import sys
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable

# Add cutlass python path for pycute import
cutlass_python_path = Path(__file__).parent.parent / "cutlass" / "python"
sys.path.append(str(cutlass_python_path))

from pycute.int_tuple import (
    product,
    idx2crd,
    crd2idx,
    crd2crd,
    slice_,
)
from pycute.layout import (
    Layout,
    coalesce,
)

__all__ = [
    "product",
    "idx2crd",
    "crd2idx",
    "crd2crd",
    "coalesce",
    "slice_",
    "Layout",
    "visualize_layout",
]


def default_color_map(index: int) -> tuple[float, float, float]:
    # https://github.com/NVIDIA/cutlass/blob/main/include/cute/util/print_latex.hpp
    colors = plt.cm.tab20c.colors[:16]
    return colors[index % len(colors)]


def default_label_map(index: int) -> str:
    return str(index)


def visualize_layout(
    layout: Layout,
    color_map: Callable[[int], object] | None = None,
    label_map: Callable[[int], str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    # https://github.com/NVIDIA/cutlass/blob/main/include/cute/util/print_latex.hpp

    if color_map is None:
        color_map = default_color_map
    if label_map is None:
        label_map = default_label_map

    if isinstance(layout.shape, int):
        M = layout.shape
        N = 1
    elif len(layout.shape) == 1:
        M = product(layout.shape[0])
        N = 1
    elif len(layout.shape) == 2:
        M = product(layout.shape[0])
        N = product(layout.shape[1])
    else:
        raise NotImplementedError

    # Create figure and axis
    fig, ax = plt.subplots(**kwargs)

    for m in range(M):
        for n in range(N):
            if isinstance(layout.shape, int):
                index = layout(m)
            elif len(layout.shape) == 1:
                index = layout(m)
            elif len(layout.shape) == 2:
                index = layout(m, n)
            else:
                raise NotImplementedError

            # Get color and label for this index
            color = color_map(index)
            label = label_map(index)

            # Draw rectangle with color
            rect = plt.Rectangle(
                (n, M - m - 1),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Add index text in center of cell
            ax.text(
                n + 0.5,
                M - m - 0.5,
                label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

    # Add row labels (m indices) - positioned to the left
    for m in range(M):
        ax.text(
            -0.3,
            M - m - 0.5,
            str(m),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Add column labels (n indices) - positioned at the top
    for n in range(N):
        ax.text(
            n + 0.5,
            M + 0.3,
            str(n),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlim(-0.8, N + 0.8)
    ax.set_ylim(-0.8, M + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig, ax
