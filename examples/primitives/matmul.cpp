/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

#define SRC_DT int8_t
#define BIAS_DT uint16_t
#define DST_DT uint16_t

void matmul_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim MB = 3, // batch size
            M = 128, K = 256, N = 512;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = { M, K};
    memory::dims weights_dims = { K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = { M, N};

    //Prepare scale.
    // const int src_mask = 2;
    const int weight_mask = 2;
    const int dst_mask = 0;

    // Allocate buffers.
    std::vector<SRC_DT> src_data(product(src_dims));
    std::vector<SRC_DT> weights_data(product(weights_dims));
    std::vector<BIAS_DT> bias_data(product(bias_dims));
    std::vector<DST_DT> dst_data(product(dst_dims));

    // Initialize src, weights, bias.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 1;
        return static_cast<SRC_DT>(i);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 2;
        return static_cast<BIAS_DT>(i);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 3;
        return static_cast<DST_DT>(i);
    });

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::bf16, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);
    auto binary_add_md = memory::desc(src_dims, dt::s8, tag::ab);
    auto src_mem = memory(src_md, engine);
    auto weights_mem = memory(weights_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_mem = memory(dst_md, engine);
    auto binary_add_mem = memory(binary_add_md, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    matmul_ops.append_binary(algorithm::binary_add, binary_add_md);
    // matmul_ops.append_sum(2.f, 0, dnnl::memory::data_type::bf16);
    matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);

    // matmul_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1<<1);
    auto src_scale_md = memory::desc({M}, dt::f32, tag::x);
    auto wei_scale_md = memory::desc({N}, dt::f32, tag::x);
    auto src_scale_mem = memory(src_scale_md, engine);
    auto wei_scale_mem = memory(wei_scale_md, engine);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(
            engine, src_md, weights_md, bias_md, dst_md, matmul_attr);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
            binary_add_mem});
    // matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem});
    matmul_args.insert(
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
