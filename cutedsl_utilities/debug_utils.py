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
def print_tensor(tensor: cute.Tensor | cute.TensorSSA, tid: int = 0, bid: int = 0) -> None:
    if thread(tid=tid, bid=bid):
        if cutlass.const_expr(isinstance(tensor, cute.Tensor)):
            cute.print_tensor(tensor)
        elif cutlass.const_expr(isinstance(tensor, cute.TensorSSA)):
            tensor_rmem = cute.make_fragment(
                layout_or_shape=tensor.shape,
                dtype=tensor.dtype)
            tensor_rmem.store(tensor)
            cute.print_tensor(tensor_rmem)
        else:
            raise NotImplementedError
