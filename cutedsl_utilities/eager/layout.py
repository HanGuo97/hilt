import cutlass
import cutlass.cute as cute
import matplotlib.pyplot as plt
from typing import Any, NamedTuple, Self
from cutedsl_utilities.pycute_utils import (
    visualize_layout,
    Layout as PyCuTeLayout,
)

from .base import CuTeEager


class LayoutStruct(NamedTuple):
    shape: tuple | cute.Shape
    stride: tuple

    @classmethod
    def from_layout(
        cls: type[Self],
        layout: cute.Layout,
    ) -> Self:
        assert isinstance(layout, cute.Layout)
        return cls(shape=layout.shape, stride=layout.stride)


class CuTeLayout(CuTeEager):

    def __init__(
        self,
        shape: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
    ) -> None:
        self._shape = shape
        self._stride = stride

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def stride(self) -> tuple | None:
        return self._stride

    @property
    def layout(self) -> PyCuTeLayout:
        return PyCuTeLayout(
            _shape=self.shape,
            _stride=self.stride,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "stride": self.stride,
        }

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return LayoutStruct

    @classmethod
    def from_struct(cls: type[Self], struct: LayoutStruct) -> Self:
        assert isinstance(struct, cls.get_struct_type())
        return cls(**struct._asdict())

    def to_struct(self) -> LayoutStruct:
        struct_cls = self.get_struct_type()
        metadata = self.get_metadata()
        return struct_cls(**metadata)

    def visualize(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        return visualize_layout(layout=self.layout, **kwargs)
