import math
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
    "log2",
    "log",
]

LOGE_2 = math.log(2.0)


def make_dispatch_function(
    fn_tensor: Callable[[object], cute.Tensor] | None = None,
    fn_tensorssa: Callable[[object], cute.TensorSSA] | None = None,
    fn_scalar: Callable[[object], Scalar] | None = None,
    dispatch_policy: str | None = None,
) -> Callable[[object], Tensor]:
    """Creates a function that dispatches to appropriate function based on input types.

    :param fn_tensor: function to call for cute.Tensor inputs
    :param fn_tensorssa: function to call for cute.TensorSSA inputs
    :param fn_scalar: function to call for scalar inputs
    :param dispatch_policy: dispatching strategy
    :return: dispatcher function
    """
    if dispatch_policy is None:
        dispatch_policy = "first"

    def _dispatcher(*args, **kwargs) -> Tensor:
        if cutlass.const_expr(dispatch_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                dispatch_arg = args[0]
            else:
                raise ValueError

        if cutlass.const_expr(isinstance(dispatch_arg, cute.Tensor)):
            if cutlass.const_expr(fn_tensor is not None):
                return fn_tensor(*args, **kwargs)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(dispatch_arg, cute.TensorSSA)):
            if cutlass.const_expr(fn_tensorssa is not None):
                return fn_tensorssa(*args, **kwargs)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(dispatch_arg, Scalar)):
            if cutlass.const_expr(fn_scalar is not None):
                return fn_scalar(*args, **kwargs)
            else:
                raise NotImplementedError
        raise NotImplementedError

    return _dispatcher


def make_tensorssa_fn_from_scalar_fn(
    fn_scalar: Callable[[object], Scalar],
    variadic_policy: str | None = None,
) -> Callable[[object], cute.TensorSSA]:
    # https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

    if variadic_policy is None:
        variadic_policy = "first"

    @cute.jit
    def _tensorssa_fn(*args, **kwargs) -> cute.TensorSSA:
        if cutlass.const_expr(variadic_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                x = args[0]
                extra_args = args[1:]
            else:
                raise ValueError

        assert cutlass.const_expr(isinstance(x, cute.TensorSSA))
        res = cute.make_fragment(x.shape, x.dtype)
        res.store(x)

        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = fn_scalar(res[i], *extra_args, **kwargs)

        return res.load()

    return _tensorssa_fn


def make_tensorssa_fn_from_scalar_fn_different_dtype(
    fn_scalar: Callable[[object], Scalar],
    dtype: type[cute.Numeric],
    variadic_policy: str | None = None,
) -> Callable[[object], cute.TensorSSA]:
    # https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

    if variadic_policy is None:
        variadic_policy = "first"

    @cute.jit
    def _tensorssa_fn(*args, **kwargs) -> cute.TensorSSA:
        if cutlass.const_expr(variadic_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                x = args[0]
                extra_args = args[1:]
            else:
                raise ValueError

        assert cutlass.const_expr(isinstance(x, cute.TensorSSA))
        tensor_x = cute.make_fragment(x.shape, x.dtype)
        tensor_y = cute.make_fragment(x.shape, dtype)
        tensor_x.store(x)

        for i in cutlass.range_constexpr(cute.size(x.shape)):
            tensor_y[i] = fn_scalar(tensor_x[i], *extra_args, **kwargs)

        return tensor_y.load()

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
def _rsqrt(a: float | cute.Float32, *, loc=None, ip=None) -> cute.Float32:
    # https://github.com/Dao-AILab/quack/blob/main/quack/utils.py
    assert cutlass.const_expr(isinstance(a, float | cute.Float32))
    return cute.Float32(
        llvm.inline_asm(
            T.f32(),
            [cute.Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


rsqrt = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(_rsqrt),
    fn_scalar=_rsqrt,
)


@dsl_user_op
def _log2(a: float | cute.Float32, *, loc=None, ip=None) -> cute.Float32:
    # https://github.com/Dao-AILab/quack/blob/main/quack/utils.py
    assert cutlass.const_expr(isinstance(a, float | cute.Float32))
    return cute.Float32(
        llvm.inline_asm(
            T.f32(),
            [cute.Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _log(a: float | cute.Float32, *, loc=None, ip=None) -> cute.Float32:
    # https://github.com/Dao-AILab/quack/blob/main/quack/cross_entropy.py
    return _log2(a, loc=loc, ip=ip) * LOGE_2


log2 = make_dispatch_function(
    fn_tensorssa=cute.math.log2,
    fn_scalar=_log2,
)


log = make_dispatch_function(
    fn_tensorssa=cute.math.log,
    fn_scalar=_log,
)
