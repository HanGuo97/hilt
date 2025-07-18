# https://github.com/pytorch/extension-cpp

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

LIBRARY_NAME = "rapier"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
            "-std=c++17",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
        ],
    }


    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "csrc")
    sources = (
        list(glob.glob(os.path.join(extensions_dir, "*.cpp"))) +
        list(glob.glob(os.path.join(extensions_dir, "*.cu"))))

    include_dirs = [
        os.path.join(this_dir, "cutlass/include"),
        os.path.join(this_dir, "cutlass/tools/util/include"),
    ]

    ext_modules = [
        CUDAExtension(
            f"{LIBRARY_NAME}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
            include_dirs=include_dirs,
        )
    ]

    return ext_modules


setup(
    name=LIBRARY_NAME,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="CuTeDSL Utilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HanGuo97/rapier",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
