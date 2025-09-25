import functools
import cutlass
import cutlass.cute as cute
import torch.utils._pytree as pytree
from typing import Any, Callable, cast
from cutlass._mlir.dialects import cute as _cute_ir

from .base import (
    CuTeEager,
)
from .tensor import (
    TensorStruct,
    NumericStruct,
    TensorSSAStruct,
    ArithValueStruct,
    CoordinateTensorStruct,
    TensorStructWithPointer,
    SSATensor,
    NumericTensor,
    ReferenceTensor,
    ArithValueTensor,
    CoordinateTensor,
)
from .layout import (
    LayoutStruct,
    CuTeLayout,
)

CuTeEagerStruct = (
    LayoutStruct |
    TensorStruct |
    NumericStruct |
    TensorSSAStruct |
    ArithValueStruct |
    CoordinateTensorStruct |
    TensorStructWithPointer
)

CuTeEagerClasses = [
    ReferenceTensor,
    CoordinateTensor,
    SSATensor,
    ArithValueTensor,
    NumericTensor,
    CuTeLayout,
]


def create_tensor_from_tensor_struct(tree: pytree.PyTree) -> pytree.PyTree:
    def _fn(maybe_tensor: Any) -> Any:
        if cutlass.const_expr(isinstance(maybe_tensor, TensorStruct)):
            raise NotImplementedError
        if cutlass.const_expr(isinstance(maybe_tensor, CoordinateTensorStruct)):
            struct = cast(CoordinateTensorStruct, maybe_tensor)
            layout = cute.make_layout(shape=struct.shape, stride=struct.stride)
            return cute.make_tensor(iterator=struct.iterator, layout=layout)
        if cutlass.const_expr(isinstance(maybe_tensor, TensorStructWithPointer)):
            struct = cast(TensorStructWithPointer, maybe_tensor)
            layout = cute.make_layout(shape=struct.shape, stride=struct.stride)
            iterator = cute.core.make_ptr(
                dtype=struct.dtype,
                value=struct.pointer,
                mem_space=cute.AddressSpace[struct.memspace],
                assumed_align=struct.alignment,
            )
            return cute.make_tensor(iterator=iterator, layout=layout)
        if cutlass.const_expr(isinstance(maybe_tensor, TensorSSAStruct)):
            struct = cast(TensorSSAStruct, maybe_tensor)
            tensor = cute.make_fragment(layout_or_shape=struct.shape, dtype=struct.dtype)
            return tensor.load()
        if cutlass.const_expr(isinstance(maybe_tensor, ArithValueStruct)):
            struct = cast(ArithValueStruct, maybe_tensor)
            return cutlass.cutlass_dsl.arith.constant(result=struct.dtype.mlir_type, value=struct.dtype.zero)
        if cutlass.const_expr(isinstance(maybe_tensor, NumericStruct)):
            struct = cast(NumericStruct, maybe_tensor)
            return struct.dtype.zero
        if cutlass.const_expr(isinstance(maybe_tensor, LayoutStruct)):
            struct = cast(LayoutStruct, maybe_tensor)
            return cute.make_layout(shape=struct.shape, stride=struct.stride)
        return maybe_tensor

    return pytree.tree_map(func=_fn, tree=tree, is_leaf=lambda node: isinstance(node, CuTeEagerStruct))


def create_tensor_struct_from_tensor(tree: pytree.PyTree, constant_as_numeric: bool) -> pytree.PyTree:
    def _fn(maybe_tensor: Any) -> Any:
        if cutlass.const_expr(isinstance(maybe_tensor, cute.Tensor)):
            assert cutlass.const_expr(isinstance(maybe_tensor, cute.core._Tensor))
            tensor = cast(cute.core._Tensor, maybe_tensor)
            if cutlass.const_expr(isinstance(tensor.type, _cute_ir.MemRefType)):
                return TensorStruct.from_tensor(tensor)
            if cutlass.const_expr(isinstance(tensor.type, _cute_ir.CoordTensorType)):
                return CoordinateTensorStruct.from_tensor(tensor)
            raise NotImplementedError
        if cutlass.const_expr(isinstance(maybe_tensor, cute.TensorSSA)):
            return TensorSSAStruct.from_tensor(maybe_tensor)
        if cutlass.const_expr(isinstance(maybe_tensor, cutlass.cutlass_dsl.cutlass_arith.ArithValue)):
            return ArithValueStruct.from_tensor(maybe_tensor)
        if cutlass.const_expr(isinstance(maybe_tensor, cute.Numeric)):
            return NumericStruct.from_tensor(maybe_tensor)
        if cutlass.const_expr(isinstance(maybe_tensor, int | float | bool) and constant_as_numeric):
            if cutlass.const_expr(isinstance(maybe_tensor, int)):
                numeric = cute.Int64(maybe_tensor)
            elif cutlass.const_expr(isinstance(maybe_tensor, float)):
                numeric = cute.Float32(maybe_tensor)
            elif cutlass.const_expr(isinstance(maybe_tensor, bool)):
                numeric = cute.Boolean(maybe_tensor)
            else:
                raise NotImplementedError
            return NumericStruct.from_tensor(numeric)
        if cutlass.const_expr(isinstance(maybe_tensor, cute.Layout)):
            return LayoutStruct.from_layout(maybe_tensor)
        return maybe_tensor
    return pytree.tree_map(func=_fn, tree=tree)


def create_tensor_struct_from_rapier_tensor(tree: pytree.PyTree) -> pytree.PyTree:
    def _fn(maybe_tensor: Any) -> Any:
        if cutlass.const_expr(isinstance(maybe_tensor, CuTeEager)):
            tensor = cast(CuTeEager, maybe_tensor)
            struct = tensor.to_struct()
            if isinstance(struct, TensorStruct):
                assert isinstance(tensor, ReferenceTensor)
                pointer = tensor.make_pointer()
                pointer = cast(cute.runtime._Pointer, pointer)
                struct = TensorStructWithPointer(
                    pointer=pointer._pointer,
                    **struct._asdict(),
                )
            return struct
        return maybe_tensor
    return pytree.tree_map(func=_fn, tree=tree)


def create_rapier_tensor_from_tensor_struct(tree: pytree.PyTree) -> pytree.PyTree:
    def _fn(maybe_tensor: Any) -> Any:
        for cls in CuTeEagerClasses:
            if cutlass.const_expr(isinstance(maybe_tensor, cls.get_struct_type())):
                return cls.from_struct(maybe_tensor)

        return maybe_tensor

    return pytree.tree_map(func=_fn, tree=tree, is_leaf=lambda node: isinstance(node, CuTeEagerStruct))


def cute_apply(*, raw: bool = False, constant_as_numeric: bool = False) -> Callable:
    """
    Decorator that bridges RapierTensor API with CUTLASS/cute JIT execution.

    Usage:
    @cute_apply()
    def my_function(...): ...

    @cute_apply(raw=True)
    def my_function(...): ...

    Transforms functions operating on cute objects to work with RapierTensor instances.
    Handles serialization: RapierTensor -> struct -> JIT execution -> struct -> RapierTensor.

    Args:
        raw: If True, return the raw outputs without wrapping them in `RapierTensor`
        constant_as_numeric: If True, treat constants as numeric values

    Returns:
        Decorator function
    """

    def decorator(fn: Callable) -> Callable:

        @cute.jit
        def _cute_apply(
            serialized_args: cutlass.Constexpr,
            serialized_kwargs: cutlass.Constexpr,
        ) -> pytree.PyTree:
            cute_args, cute_kwargs = create_tensor_from_tensor_struct((serialized_args, serialized_kwargs))
            output = fn(*cute_args, **cute_kwargs)
            return create_tensor_struct_from_tensor(output, constant_as_numeric=constant_as_numeric)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> pytree.PyTree:
            serialized_args, serialized_kwargs = create_tensor_struct_from_rapier_tensor((args, kwargs))
            serialized_outputs = _cute_apply(
                serialized_args=serialized_args,
                serialized_kwargs=serialized_kwargs,
            )

            if raw:
                return serialized_outputs
            else:
                return create_rapier_tensor_from_tensor_struct(serialized_outputs)

        return wrapper

    return decorator
