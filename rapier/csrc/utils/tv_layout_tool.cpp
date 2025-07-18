#include <string>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <vector>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "cute/tensor.hpp"
using namespace cute;

using Element = cute::half_t;


extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

// Helper struct to hold layout parameters
struct LayoutParams {
    int64_t tiler_m     , tiler_n;
    int64_t thr_shape_m , thr_shape_n;
    int64_t val_shape_m , val_shape_n;
    int64_t thr_stride_m, thr_stride_n;
    int64_t val_stride_m, val_stride_n;
};

// Create and validate layout parameters
LayoutParams create_layout_params(
    int64_t tiler_m     , int64_t tiler_n,
    int64_t thr_shape_m , int64_t thr_shape_n,
    int64_t val_shape_m , int64_t val_shape_n,
    int64_t thr_stride_m, int64_t thr_stride_n,
    int64_t val_stride_m, int64_t val_stride_n
) {
    // Input validation
    if (tiler_m     <= 0 || tiler_n     <= 0 ||
        thr_shape_m <= 0 || thr_shape_n <= 0 ||
        val_shape_m <= 0 || val_shape_n <= 0 ||
        thr_stride_m < 0 || thr_stride_n < 0 ||
        val_stride_m < 0 || val_stride_n < 0)
    {
        throw std::invalid_argument("Invalid parameters");
    }
    return {tiler_m     , tiler_n,
            thr_shape_m , thr_shape_n,
            val_shape_m , val_shape_n,
            thr_stride_m, thr_stride_n,
            val_stride_m, val_stride_n};
}

// Generate layout configuration message
std::string generate_layout_message(const LayoutParams& params, const std::exception* e = nullptr) {
    std::ostringstream oss;

    // Always print configuration
    oss << "% Layout Configuration:\n"
        << "% Tiler: (" << params.tiler_m << ", " << params.tiler_n << ")\n"
        << "% Thread Shape: (" << params.thr_shape_m << ", " << params.thr_shape_n << ")\n"
        << "% Thread Stride: (" << params.thr_stride_m << ", " << params.thr_stride_n << ")\n"
        << "% Value Shape: (" << params.val_shape_m << ", " << params.val_shape_n << ")\n"
        << "% Value Stride: (" << params.val_stride_m << ", " << params.val_stride_n << ")\n"
        << "% Copy Size: 128 bits (16 bytes)\n";

    // Add error message if provided
    if (e) {
        oss << "% Error generating tiled copy layout: " << e->what() << "\n";
    }
    
    return oss.str();
}

// Create CuTe tiled copy and generate LaTeX
std::string create_tiled_copy_latex(const LayoutParams& params) {

    // Create tiler layout
    auto tiler_mn = make_layout(
        make_shape(params.tiler_m, params.tiler_n)
    );

    // Create TV layout by combining thread and value layouts
    auto layout_tv = make_layout(
        make_shape (make_shape (params.thr_shape_m , params.thr_shape_n ),
                    make_shape (params.val_shape_m , params.val_shape_n )),
        make_stride(make_stride(params.thr_stride_m, params.thr_stride_n),
                    make_stride(params.val_stride_m, params.val_stride_n))
    );

    // Create copy atom with 128-bit (16 bytes) copy operation
    using CopyOp = UniversalCopy<uint_byte_t<16>>;
    using Atom = Copy_Atom<CopyOp, Element>;
    auto tiled_copy = make_tiled_copy_impl(Atom{}, tiler_mn, layout_tv);

    // Capture LaTeX output
    std::ostringstream oss;
    std::streambuf* orig = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());

    print_latex(tiled_copy);
    
    std::cout.rdbuf(orig);
    return oss.str();
}

// Main function to visualize TV layout with separate parameters
std::string visualize_layout_tv(
    int64_t tiler_m     , int64_t tiler_n,
    int64_t thr_shape_m , int64_t thr_shape_n,
    int64_t val_shape_m , int64_t val_shape_n,
    int64_t thr_stride_m, int64_t thr_stride_n,
    int64_t val_stride_m, int64_t val_stride_n
) {
    try {
        auto params = create_layout_params(
            tiler_m     , tiler_n,
            thr_shape_m , thr_shape_n,
            val_shape_m , val_shape_n,
            thr_stride_m, thr_stride_n,
            val_stride_m, val_stride_n
        );
        auto latex_output = create_tiled_copy_latex(params);
        auto config_message = generate_layout_message(params);
        return config_message + latex_output;
    } catch (const std::exception& e) {
        auto params = create_layout_params(
            tiler_m     , tiler_n,
            thr_shape_m , thr_shape_n,
            val_shape_m , val_shape_n,
            thr_stride_m, thr_stride_n,
            val_stride_m, val_stride_n
        );
        return generate_layout_message(params, &e);
    }
}


std::string visualize_layout_tv_impl(
    int64_t tiler_m     , int64_t tiler_n,
    int64_t thr_shape_m , int64_t thr_shape_n,
    int64_t val_shape_m , int64_t val_shape_n,
    int64_t thr_stride_m, int64_t thr_stride_n,
    int64_t val_stride_m, int64_t val_stride_n
) {
    return visualize_layout_tv(
        tiler_m     , tiler_n,
        thr_shape_m , thr_shape_n,
        val_shape_m , val_shape_n,
        thr_stride_m, thr_stride_n,
        val_stride_m, val_stride_n
    );
}

// Defines the operators
TORCH_LIBRARY(rapier, m) {
    m.def("visualize_layout_tv(int tiler_m, int tiler_n, int thr_shape_m, int thr_shape_n, int val_shape_m, int val_shape_n, int thr_stride_m, int thr_stride_n, int val_stride_m, int val_stride_n) -> str", visualize_layout_tv_impl);
}
