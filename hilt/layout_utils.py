import cutlass
import cutlass.cute as cute

__all__ = [
    "idx2crd",
]


@cute.jit
def idx2crd(idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    # https://github.com/NVIDIA/cutlass/issues/2473
    id_layout = cute.make_identity_layout(shape)
    return id_layout(idx)
