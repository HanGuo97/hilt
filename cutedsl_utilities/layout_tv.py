import os
import torch
import tempfile
import subprocess
from jinja2 import Environment
from torch.utils.cpp_extension import load_inline


def visualize_layout_tv(
    tiler_mn: tuple[int, int],
    layout_tv: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]],
    dtype: torch.dtype,
    template_path: str | None = None,
    cutlass_path: str | None = None,
    inline: bool = False,
) -> tuple[str, str]:
    """
    Visualize layout using CuTe tiled copy and generate LaTeX output.

    Args:
        tiler_mn: Tiler dimensions (m, n)
        layout_tv: Thread-value layout specification
        element_type: CuTe element type (e.g., "cute::half_t")
        copy_bytes: Copy operation size in bytes (e.g., 16)
        template_path: Optional path to the template file
        cutlass_base_dir: Optional CUTLASS base directory

    Returns:
        LaTeX representation of the Layout TV and rendered CPP source.
    """

    if dtype == torch.float16:
        element_type = "cute::half_t"
        copy_bytes = 16
    elif dtype == torch.bfloat16:
        element_type = "cute::bfloat16_t"
        copy_bytes = 16
    elif dtype == torch.float32:
        element_type = "float"
        copy_bytes = 32
    else:
        raise NotImplementedError

    if template_path is None:
        template_path = os.path.join(
            os.path.dirname(__file__),
            "csrc/utils/tv_layout_tool.cpp",
        )

    if cutlass_path is None:
        cutlass_path = os.path.join(
            os.path.dirname(__file__),
            "../cutlass/",
        )

    tiler_m, tiler_n = tiler_mn
    (
        (
            (thr_shape_m, thr_shape_n),
            (val_shape_m, val_shape_n),
        ),
        (
            (thr_stride_m, thr_stride_n),
            (val_stride_m, val_stride_n),
        ),
    ) = layout_tv

    with open(template_path) as f:
        template_content = f.read()
    environment = Environment()
    template = environment.from_string(template_content)
    cpp_source = template.render(
        element_type=element_type,
        tiler_m=tiler_m,
        tiler_n=tiler_n,
        thr_shape_m=thr_shape_m,
        thr_shape_n=thr_shape_n,
        val_shape_m=val_shape_m,
        val_shape_n=val_shape_n,
        thr_stride_m=thr_stride_m,
        thr_stride_n=thr_stride_n,
        val_stride_m=val_stride_m,
        val_stride_n=val_stride_n,
        copy_bytes=copy_bytes,
    )

    include_paths = [
        os.path.join(cutlass_path, "include"),
        os.path.join(cutlass_path, "tools/util/include"),
    ]

    if inline:
        # JIT compile using PyTorch
        module = load_inline(
            name="tv_layout_tool",
            cpp_sources=cpp_source,
            functions=["visualize_layout_tv"],
            extra_include_paths=include_paths,
            with_cuda=True,
        )
        output = module.visualize_layout_tv()
    else:
        output = compile_and_run(
            code=cpp_source,
            include_paths=include_paths,
        )
    return output, cpp_source


def compile_and_run(code: str, include_paths: list[str]) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save source code to temporary file
        source_file = os.path.join(temp_dir, "temp.cpp")
        executable_file = os.path.join(temp_dir, "temp")

        with open(source_file, "w") as f:
            f.write(code)

        # Build nvcc compile command
        compile_cmd = ["nvcc", "-std=c++17", source_file, "-o", executable_file]
        for include_path in include_paths:
            compile_cmd.extend(["-I", include_path])
        try:
            subprocess.run(
                compile_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compilation failed: {e.stderr}")

        try:
            output = subprocess.run(
                [executable_file],
                check=True,
                capture_output=True,
                text=True,
            )
            return output.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Execution failed: {e.stderr}")
