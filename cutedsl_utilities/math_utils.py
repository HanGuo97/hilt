import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from collections.abc import Callable

Scalar = cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue
Tensor = cute.Tensor | cute.TensorSSA | Scalar

__all__ = [
    "exp2",
    "exp",
    "rsqrt",
]


def make_dispatch_function(
    fn_tensor: Callable[[cute.Tensor], cute.Tensor] | None = None,
    fn_tensorssa: Callable[[cute.TensorSSA], cute.TensorSSA] | None = None,
    fn_scalar: Callable[[Scalar], Scalar] | None = None,
) -> Callable[[Tensor], Tensor]:
    """Creates a function that dispatches to appropriate function based on input type.
    
    :param fn_tensor: function to call for cute.Tensor inputs
    :param fn_tensorssa: function to call for cute.TensorSSA inputs  
    :param fn_scalar: function to call for scalar inputs
    :return: dispatcher function
    """

    def _dispatcher(x: Tensor) -> Tensor:
        if cutlass.const_expr(isinstance(x, cute.Tensor)):
            if cutlass.const_expr(fn_tensor is not None):
                return fn_tensor(x)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
            if cutlass.const_expr(fn_tensorssa is not None):
                return fn_tensorssa(x)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(x, Scalar)):
            if cutlass.const_expr(fn_scalar is not None):
                return fn_scalar(x)
            else:
                raise NotImplementedError
        raise NotImplementedError

    return _dispatcher


def make_tensorssa_fn_from_scalar_fn(
    fn_scalar: Callable[[Scalar], Scalar],
) -> Callable[[cute.TensorSSA], cute.TensorSSA]:
    # https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
    def _tensorssa_fn(x: cute.TensorSSA) -> cute.TensorSSA:
        res = cute.make_fragment(x.shape, x.dtype)
        res.store(x)

        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = fn_scalar(res[i])

        return res.load()

    return _tensorssa_fn


exp2 = make_dispatch_function(
    fn_tensorssa=cute.math.exp2,
    fn_scalar=cute.arch.exp2,
)


exp = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(cute.arch.exp),
    fn_scalar=cute.arch.exp,
)


@dsl_user_op
def rsqrt(a: float | cute.Float32, *, loc=None, ip=None) -> cute.Float32:
    assert cutlass.const_expr(isinstance(a, float | cute.Float32))
    return cute.Float32(
        llvm.inline_asm(
            T.f32(),
            [cute.Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


rsqrt = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(rsqrt),
    fn_scalar=rsqrt,
)
