import torch
import cutlass
import cutlass.torch
import cutlass.cute as cute
from typing import Any, NamedTuple, Self
from ..dtype_utils import get_dtype
from ..pycute_utils import product
from .base import CuTeEager

CuTeTensorType = cute.runtime._Tensor | cute.core._Tensor

TORCH_DTYPE_TO_CUTLASS_DTYPE_MAP = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


class TensorStruct(NamedTuple):
    shape: tuple
    stride: tuple
    dtype: type[cute.Numeric]
    memspace: str
    alignment: int | None

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: cute.Tensor,
    ) -> Self:
        assert isinstance(tensor, CuTeTensorType)
        assert isinstance(tensor.memspace, cute.AddressSpace)
        if isinstance(tensor.iterator, cute.runtime._Pointer):
            alignment = tensor.iterator._assumed_align
        elif isinstance(tensor.iterator, cute.core._Pointer):
            alignment = tensor.iterator.alignment
        else:
            raise TypeError
        return cls(
            shape=tensor.shape,
            stride=tensor.stride,
            dtype=tensor.element_type,
            memspace=tensor.memspace.name,
            alignment=alignment,
        )


class CoordinateTensorStruct(NamedTuple):
    iterator: tuple
    shape: tuple
    stride: tuple

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: cute.Tensor,
    ) -> Self:
        assert isinstance(tensor, cute.core._Tensor)
        assert isinstance(tensor.iterator, tuple)
        return cls(
            iterator=tensor.iterator,
            shape=tensor.shape,
            stride=tensor.stride,
        )


class TensorStructWithPointer(NamedTuple):
    pointer: int
    shape: tuple
    stride: tuple
    dtype: type[cute.Numeric]
    memspace: str
    alignment: int | None


class TensorSSAStruct(NamedTuple):
    shape: tuple | cute.Shape
    dtype: type[cute.Numeric]

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: cute.TensorSSA,
    ) -> Self:
        assert isinstance(tensor, cute.TensorSSA)
        return cls(shape=tensor.shape, dtype=tensor.dtype)


class ArithValueStruct(NamedTuple):
    dtype: type[cute.Numeric]

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: cutlass.cutlass_dsl.cutlass_arith.ArithValue,
    ) -> Self:
        assert isinstance(tensor, cutlass.cutlass_dsl.cutlass_arith.ArithValue)
        cdtype = get_dtype(tensor)
        return cls(dtype=cdtype)


class NumericStruct(NamedTuple):
    dtype: type[cute.Numeric]

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: cute.Numeric,
    ) -> Self:
        assert isinstance(tensor, cute.Numeric)
        return cls(dtype=tensor.dtype)


class CuTeTensor(CuTeEager):

    def __init__(
        self,
        shape: tuple,
        stride: tuple | None,
        dtype: type[cute.Numeric],
    ) -> None:
        self._shape = shape
        self._stride = stride
        self._dtype = dtype

    @property
    def shape(self) -> tuple:
        return self._shape

    def size(self) -> tuple:
        return self.shape

    def stride(self) -> tuple | None:
        return self._stride

    @property
    def dtype(self) -> type[cute.Numeric]:
        return self._dtype

    @property
    def torch_dtype(self) -> torch.dtype:
        return cutlass.torch.dtype(self.dtype)

    @property
    def ndim(self) -> int:
        return self.dim()

    def numel(self) -> int:
        return product(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def get_metadata(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        raise NotImplementedError


class ReferenceTensor(CuTeTensor):

    def __init__(
        self,
        shape: tuple,
        stride: tuple,
        dtype: type[cute.Numeric],
        memspace: str,
        alignment: int | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            stride=stride,
            dtype=dtype,
        )
        self._memspace = memspace
        self._alignment = alignment

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "stride": self.stride(),
            "dtype": self.dtype,
            "memspace": self.memspace,
            "alignment": self.alignment,
        }

    @property
    def memspace(self) -> str:
        return self._memspace

    @property
    def alignment(self) -> int | None:
        return self._alignment

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return TensorStruct

    def make_pointer(self) -> cute.Pointer:
        if self.alignment is not None and self.alignment > 1:
            # Allocate buffer with extra space for alignment
            extra_elements = (self.alignment - 1) // self.torch_dtype.itemsize
            data = torch.empty(
                self.numel() + extra_elements,
                dtype=self.torch_dtype,
            )
            data_ptr = data.data_ptr()
            data_ptr = ((data_ptr + self.alignment - 1) // self.alignment) * self.alignment
        else:
            data = torch.empty_strided(
                size=self.size(),
                stride=self.stride(),
                dtype=self.torch_dtype,
            )
            data_ptr = data.data_ptr()

        # store buffer to prevent GC
        self._data = data
        return cute.runtime.make_ptr(
            dtype=self.dtype,
            value=data_ptr,
            mem_space=cute.AddressSpace[self.memspace],
            assumed_align=self.alignment,
        )

    def load(self) -> "SSATensor":
        return SSATensor(
            shape=self.shape,
            dtype=self.dtype,
        )


class CoordinateTensor(CuTeTensor):

    def __init__(
        self,
        iterator: tuple,
        shape: tuple,
        stride: tuple,
    ) -> None:
        super().__init__(
            shape=shape,
            stride=stride,
            dtype=cute.Int64,
        )
        self._iterator = iterator

    def get_metadata(self) -> dict[str, Any]:
        return {
            "iterator": self.iterator,
            "shape": self.shape,
            "stride": self.stride(),
        }

    @property
    def iterator(self) -> tuple:
        return self._iterator

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return CoordinateTensorStruct


class SSATensor(CuTeTensor):

    def __init__(
        self,
        shape: tuple,
        dtype: type[cute.Numeric],
    ) -> None:
        super().__init__(
            shape=shape,
            stride=None,
            dtype=dtype,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return TensorSSAStruct


class ArithValueTensor(CuTeTensor):

    def __init__(self, dtype: type[cute.Numeric]) -> None:
        super().__init__(
            shape=(),
            stride=None,
            dtype=dtype,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
        }

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return ArithValueStruct


class NumericTensor(CuTeTensor):

    def __init__(self, dtype: type[cute.Numeric]) -> None:
        super().__init__(
            shape=(),
            stride=None,
            dtype=dtype,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
        }

    @classmethod
    def get_struct_type(cls: type[Self]) -> type:
        return NumericStruct


def from_torch(
    data: torch.Tensor,
    memspace: str,
    alignment: int | None = None,
) -> ReferenceTensor:
    return ReferenceTensor(
        shape=data.size(),
        stride=data.stride(),
        dtype=TORCH_DTYPE_TO_CUTLASS_DTYPE_MAP[data.dtype],
        memspace=memspace,
        alignment=alignment,
    )
