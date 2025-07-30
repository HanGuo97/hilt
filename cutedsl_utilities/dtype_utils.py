import ipdb
import cutlass
import cutlass.cute as cute
from typing import cast

__all__ = [
    "get_dtype",
]


def mlir_type_to_cute_type(mdtype: cutlass.cutlass_dsl.cutlass_arith.ir.Type) -> type[cute.Numeric]:
    assert isinstance(mdtype, cutlass.cutlass_dsl.cutlass_arith.ir.Type)
    if cutlass.cutlass_dsl.cutlass_arith.ir.F32Type.isinstance(mdtype):
        return cutlass.Float32
    raise NotImplementedError


def get_dtype(x: cute.Tensor | cute.TensorSSA | cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue) -> type[cute.Numeric]:
    if cutlass.const_expr(isinstance(x, cute.Tensor)):
        x = cast(cute.Tensor, x)
        return x.element_type
    if cutlass.const_expr(isinstance(x, cute.TensorSSA | cute.Numeric)):
        x = cast(cute.TensorSSA | cute.Numeric, x)
        return x.dtype
    if cutlass.const_expr(isinstance(x, cutlass.cutlass_dsl.cutlass_arith.ArithValue)):
        x = cast(cutlass.cutlass_dsl.cutlass_arith.ArithValue, x)
        return mlir_type_to_cute_type(x.type)

    raise NotImplementedError
