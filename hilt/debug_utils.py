# https://github.com/NVIDIA/cutlass/blob/main/include/cute/util/debug.hpp

import cutlass
import cutlass.cute as cute

__all__ = [
    "block",
    "thread",
    "thread0",
    "block0",
    "printf",
    "print_tensor",
    "print_tensorssa",
    "runtime_print",
]


@cute.jit
def block(bid: int) -> bool:
    bidx, bidy, bidz = cute.arch.block_idx()
    gdimx, gdimy, _ = cute.arch.grid_dim()
    current_block_id = bidx + bidy * gdimx + bidz * gdimx * gdimy
    return current_block_id == bid


@cute.jit
def thread(tid: int, bid: int) -> bool:
    tidx, tidy, tidz = cute.arch.thread_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    current_thread_id = tidx + tidy * bdimx + tidz * bdimx * bdimy
    return (current_thread_id == tid) and block(bid)


@cute.jit
def thread0() -> bool:
    return thread(0, 0)


@cute.jit
def block0() -> bool:
    return block(0)


@cute.jit
def printf(*args, tid: int = 0, bid: int = 0) -> None:
    if thread(tid=tid, bid=bid):
        cute.printf(*args)


@cute.jit
def print_tensor(tensor: cute.Tensor, tid: int = 0, bid: int = 0) -> None:
    if thread(tid=tid, bid=bid):
        cute.print_tensor(tensor)


@cute.jit
def print_tensorssa(tensor: cute.TensorSSA, tid: int = 0, bid: int = 0) -> None:
    tensor_rmem = cute.make_fragment(
        layout_or_shape=tensor.shape,
        dtype=tensor.dtype)
    tensor_rmem.store(tensor)
    print_tensor(tensor_rmem, tid=tid, bid=bid)


@cute.jit
def runtime_print(x: cute.Tensor | cute.TensorSSA | str | object, tid: int = 0, bid: int = 0) -> None:
    if cutlass.const_expr(isinstance(x, str)):
        printf(x, tid=tid, bid=bid)
    elif cutlass.const_expr(isinstance(x, cute.Tensor)):
        print_tensor(x, tid=tid, bid=bid)
    elif cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        print_tensorssa(x, tid=tid, bid=bid)
    else:
        printf(f"Type: {type(x)}\t" + "Value: {}", x, tid=tid, bid=bid)
