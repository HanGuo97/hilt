#include <string>
#include <sstream>
#include <iostream>

#include "cute/tensor.hpp"
using namespace cute;

using Element = {{ element_type }};

// Create CuTe tiled copy and generate LaTeX
std::string create_tiled_copy_latex() {

    // Create tiler layout
    auto tiler_mn = make_layout(
        make_shape(Int<{{ tiler_m }}>{}, Int<{{ tiler_n }}>{})
    );

    // Create TV layout by combining thread and value layouts
    auto layout_tv = make_layout(
        make_shape (make_shape (Int<{{ thr_shape_m  }}>{}, Int<{{ thr_shape_n  }}>{}),
                    make_shape (Int<{{ val_shape_m  }}>{}, Int<{{ val_shape_n  }}>{})),
        make_stride(make_stride(Int<{{ thr_stride_m }}>{}, Int<{{ thr_stride_n }}>{}),
                    make_stride(Int<{{ val_stride_m }}>{}, Int<{{ val_stride_n }}>{}))
    );

    // Create copy atom with configurable copy operation
    using CopyOp = UniversalCopy<uint_byte_t<{{ copy_bytes }}>>;
    using Atom = Copy_Atom<CopyOp, Element>;
    auto tiled_copy = make_tiled_copy_impl(Atom{}, layout_tv, tiler_mn);

    // Capture LaTeX output
    std::ostringstream oss;
    std::streambuf* orig = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());

    print_latex(tiled_copy);
    
    std::cout.rdbuf(orig);
    return oss.str();
}
