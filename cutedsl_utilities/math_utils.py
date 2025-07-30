import cutlass
import cutlass.cute as cute
from collections.abc import Callable

Scalar = cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue
Tensor = cute.Tensor | cute.TensorSSA | Scalar


def make_dispatch_function(
    fn_tensor: Callable[[cute.Tensor], cute.Tensor],
    fn_tensorssa: Callable[[cute.TensorSSA], cute.TensorSSA],
    fn_scalar: Callable[[Scalar], Scalar],
) -> Callable[[Tensor], Tensor]:
    """Creates a function that dispatches to appropriate function based on input type.
    
    :param fn_tensor: function to call for cute.Tensor inputs
    :param fn_tensorssa: function to call for cute.TensorSSA inputs  
    :param fn_scalar: function to call for scalar inputs
    :return: dispatcher function
    """

    def _dispatcher(x: Tensor) -> Tensor:
        if cutlass.const_expr(isinstance(x, cute.Tensor)):
            return fn_tensor(x)
        if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
            return fn_tensorssa(x)
        if cutlass.const_expr(isinstance(x, Scalar)):
            return fn_scalar(x)
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
