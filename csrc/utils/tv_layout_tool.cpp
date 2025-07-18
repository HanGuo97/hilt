#include <torch/extension.h>
#include <string>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdexcept>

#include "cute/tensor.hpp"
using namespace cute;

using Element = cute::half_t;

std::string generate_tiled_copy_layout(
    const std::tuple<std::tuple<std::tuple<int, int>, std::tuple<int, int>>, std::tuple<std::tuple<int, int>, std::tuple<int, int>>>& tv_layout,
    const std::tuple<int, int>& tiler_mn = std::make_tuple(4, 128)
) {
    auto [shapes, strides] = tv_layout;
    auto [thr_shape, val_shape] = shapes;
    auto [thr_stride, val_stride] = strides;
    auto [thr_m, thr_n] = thr_shape;
    auto [val_m, val_n] = val_shape;
    auto [stride_m, stride_n] = thr_stride;
    auto [val_stride_m, val_stride_n] = val_stride;
    auto [mn_tiler_m, mn_tiler_n] = tiler_mn;
    
    // Input validation
    if (thr_m <= 0 ||
        thr_n <= 0 ||
        val_m <= 0 ||
        val_n <= 0 ||
        mn_tiler_m <= 0 ||
        mn_tiler_n <= 0 ||
        stride_m < 0 ||
        stride_n < 0 ||
        val_stride_m < 0 ||
        val_stride_n < 0)
    {
        throw std::invalid_argument("Invalid parameters");
    }
    
    // Create runtime layouts using CuTe's dynamic layout construction
    std::stringstream ss;
    std::streambuf* orig = std::cout.rdbuf();
    std::cout.rdbuf(ss.rdbuf());
    
    try {
        // Create thread layout
        auto thread_layout = make_layout(
            make_shape(thr_m, thr_n),
            make_stride(stride_m, stride_n)
        );
        
        // Create value layout
        auto value_layout = make_layout(
            make_shape(val_m, val_n),
            make_stride(val_stride_m, val_stride_n)
        );
        
        // Create TV layout by combining thread and value layouts
        auto tv_layout_cute = make_layout(
            make_shape(get_shape(thread_layout), get_shape(value_layout)),
            make_stride(get_stride(thread_layout), get_stride(value_layout))
        );
        
        // Create tiler layout
        auto tiler_layout = make_layout(
            make_shape(mn_tiler_m, mn_tiler_n)
        );
        
        // Create copy atom with 128-bit (16 bytes) copy operation
        using CopyOp = UniversalCopy<uint_byte_t<16>>;
        using Atom = Copy_Atom<CopyOp, Element>;
        auto tiled_copy = make_tiled_copy_impl(Atom{}, tv_layout_cute, tiler_layout);
        print_latex(tiled_copy);
        
    } catch (const std::exception& e) {
        std::cout << "% Error generating tiled copy layout: " << e.what() << std::endl;
        std::cout << "% Layout Configuration:" << std::endl;
        std::cout << "% Thread Shape: (" << thr_m << ", " << thr_n << ")" << std::endl;
        std::cout << "% Thread Stride: (" << stride_m << ", " << stride_n << ")" << std::endl;
        std::cout << "% Value Shape: (" << val_m << ", " << val_n << ")" << std::endl;
        std::cout << "% Value Stride: (" << val_stride_m << ", " << val_stride_n << ")" << std::endl;
        std::cout << "% Tiler: (" << mn_tiler_m << ", " << mn_tiler_n << ")" << std::endl;
        std::cout << "% Copy Size: 128 bits (16 bytes)" << std::endl;
    }
    
    std::cout.rdbuf(orig);
    return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Rapier - CuTe TV Layout Tool";
    
    m.def("generate_tiled_copy_layout", 
          [](const std::tuple<std::tuple<std::tuple<int, int>, std::tuple<int, int>>, std::tuple<std::tuple<int, int>, std::tuple<int, int>>>& tv_layout,
             const std::tuple<int, int>& tiler_mn = std::make_tuple(4, 128)) -> std::string {
                 try {
                     return generate_tiled_copy_layout(tv_layout, tiler_mn);
                 } catch (const std::exception& e) {
                     throw std::runtime_error(e.what());
                 }
             },
          "Generate tiled copy LaTeX output from TV layout parameters",
          py::arg("tv_layout"),
          py::arg("tiler_mn") = std::make_tuple(4, 128));
}