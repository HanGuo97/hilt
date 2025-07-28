# https://github.com/NVIDIA/cutlass/blob/main/include/cute/util/debug.hpp

import cutlass.cute as cute


def block(bid: int) -> bool:
    bidx, bidy, bidz = cute.arch.block_idx()
    gdimx, gdimy, _ = cute.arch.grid_dim()
    current_block_id = bidx + bidy * gdimx + bidz * gdimx * gdimy
    return current_block_id == bid


def thread(tid: int, bid: int) -> bool:
    tidx, tidy, tidz = cute.arch.thread_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    current_thread_id = tidx + tidy * bdimx + tidz * bdimx * bdimy
    return (current_thread_id == tid) and block(bid)


def thread0() -> bool:
    return thread(0, 0)


def block0() -> bool:
    return block(0)


def printf(*args, tid: int = 0, bid: int = 0, **kwargs) -> None:
    if thread(tid=tid, bid=bid):
        cute.printf(*args, **kwargs)


def print_tensor(*args, tid: int = 0, bid: int = 0, **kwargs) -> None:
    if thread(tid=tid, bid=bid):
        cute.print_tensor(*args, **kwargs)
