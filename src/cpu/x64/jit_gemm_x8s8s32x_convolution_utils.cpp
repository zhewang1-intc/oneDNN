/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <cstdlib>
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace gemm_x8s8s32x_convolution_utils {
using namespace dnnl::impl::cpu::gemm_x8s8s32x_convolution_utils;

struct jit_pp_ker_t : pp_ker_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);

    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp) {
#define PARAM_OFF(field) offsetof(ker_args_t, field)
        if (jcp.with_eltwise || jcp.with_binary) {
            using namespace binary_injector;
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr size_t helper_vmm_idx = 31;
            // tail_size = 1 just indicates that tailing is to be performed
            // actual tail value is held in opmask passed to injector
            static constexpr size_t tail_size = 1;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                    r13, r14, preserve_gpr, preserve_vmm,
                    PARAM_OFF(post_ops_binary_rhs_arg_vec),
                    memory_desc_wrapper(pd->dst_md()), tail_size, opmask_binary,
                    use_exact_tail_scalar_bcast};
            const static_params_t static_params {
                    this->param1, rhs_arg_static_params};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<avx512_core>>(
                    this, jcp_.post_ops, static_params);
        }

        if (jcp.bias_data_type != data_type::undef)
            bias_data_type_size_ = types::data_type_size(jcp.bias_data_type);
        dst_data_type_size_ = types::data_type_size(jcp.dst_data_type);
#undef PARAM_OFF
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(void *void_dst, const acc_data_t *acc, const char *bias,
            const float *scales, float nslope, float sum_scale,
            float signed_scale, int g, size_t start, size_t end,
            const int32_t *zp_src, const int32_t *zp_dst,
            const int32_t *zp_src_comp, const void *post_ops_binary_rhs_arg_vec,
            const void *dst_orig, const exec_ctx_t & /* ctx */,
            const memory_desc_t & /* dst_md */) const override {

        if (end <= start) return;

        char *dst = (char *)void_dst;

        ker_args_t args;
        const auto dv = std::div(start, jcp_.oc);
        const size_t oc_offset = dv.rem;
        const size_t os_offset = dv.quot;
        args.acc = acc + start;
        args.dst = dst
                + (os_offset * jcp_.dst_os_stride + oc_offset)
                        * dst_data_type_size_;

        const ptrdiff_t g_oc_offset = g * jcp_.oc + oc_offset;

        args.bias = bias + g_oc_offset * bias_data_type_size_;
        args.zp_src = zp_src + (jcp_.zp.src_is_common ? 0 : g_oc_offset);
        args.zp_src_comp = zp_src_comp + g_oc_offset;
        args.zp_dst = zp_dst;
        args.scales = scales + jcp_.scale_idx_mult * g_oc_offset;
        args.nslope = nslope;
        args.sum_scale = sum_scale;
        args.signed_scale = signed_scale;
        args.len = end - start;
        args.oc_offset = oc_offset;

        args.g_oc_offset = g * jcp_.oc;
        args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
        args.dst_orig = dst_orig;
        jit_generator::operator()(&args);
    }

private:
    void apply_postops(const Xbyak::Reg64 &reg_dst, const int idx);
    void generate() override;
    void append_zp_src_comp(size_t offset, int idx, bool apply_mask);
    int vreg_dst_idx(const int idx) const noexcept;
    Xbyak::Zmm vreg_dst(int idx) const;
    Xbyak::Zmm vreg_bias(int idx) const;
    Xbyak::Zmm vreg_prev_dst(int idx) const;
    Xbyak::Zmm vreg_zp_comp_src(int idx) const;
    Xbyak::Zmm get_masked_vreg_dst(int idx, bool apply_mask) const;

    template <typename T>
    void advance_binary_postops_off(const T &offset);
    void zero_binary_postops_off();
    void set_binary_postops_off(const Xbyak::Reg64 &reg);
    const Xbyak::Opmask &opmask_binary = k2;
    const Xbyak::Reg64 &reg_tmp = rcx; // intentional for shifting purposes

    struct ker_args_t {
        char *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float nslope;
        float sum_scale;
        float signed_scale;
        size_t len;
        size_t oc_offset;
        const int32_t *zp_src;
        const int32_t *zp_dst;
        const int32_t *zp_src_comp;
        size_t g_oc_offset;
        const void *post_ops_binary_rhs_arg_vec;
        const void *dst_orig;
    };

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    size_t bias_data_type_size_ = 0;
    size_t dst_data_type_size_ = 0;
    const Xbyak::Reg64 &reg_zp_src_ = this->r13;
    const Xbyak::Reg64 &reg_zp_src_comp_ = this->r14;
    const Xbyak::Reg64 &reg_zp_dst_ = this->r15;
    const Xbyak::Opmask &kreg_rem_mask_short = k1;
    const Xbyak::Opmask &kreg_rem_mask_vlen = k3;
    static constexpr size_t def_unroll = 4u;
    size_t max_unroll = 12u;
    size_t zmm_step = 2u;

    const Xbyak::Reg64 &reg_tmp_comp
            = r12; // used to broadcast scalar values to vreg
    const Xbyak::Reg64 &reg_oc_offset = r9;
    const Xbyak::Reg64 &reg_g_oc_off = reg_tmp_comp;
    int dst_l_offset = 0;
};

template <typename T>
void jit_pp_ker_t::advance_binary_postops_off(const T &offset) {
    add(reg_g_oc_off, offset);

    Xbyak::Label end;
    cmp(reg_g_oc_off, jcp_.oc);
    jl(end, T_NEAR);
    xor_(reg_g_oc_off, reg_g_oc_off);

    L(end);
}
void jit_pp_ker_t::zero_binary_postops_off() {
    xor_(reg_g_oc_off, reg_g_oc_off);
    dst_l_offset = 0;
}
void jit_pp_ker_t::set_binary_postops_off(const Xbyak::Reg64 &reg) {
    mov(reg_g_oc_off, reg);
    dst_l_offset = 0;
}

int jit_pp_ker_t::vreg_dst_idx(const int idx) const noexcept {
    return (6 + idx * zmm_step + 0);
}

Xbyak::Zmm jit_pp_ker_t::vreg_dst(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx));
}

Xbyak::Zmm jit_pp_ker_t::vreg_bias(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx) + 1);
}

Xbyak::Zmm jit_pp_ker_t::vreg_zp_comp_src(int idx) const {
    return vreg_bias(idx);
}

Xbyak::Zmm jit_pp_ker_t::vreg_prev_dst(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx) + 2);
}

Xbyak::Zmm jit_pp_ker_t::get_masked_vreg_dst(int idx, bool apply_mask) const {
    auto vreg_dst = this->vreg_dst(idx);
    if (apply_mask)
        vreg_dst = vreg_dst | kreg_rem_mask_short;
    else
        vreg_dst = vreg_dst | kreg_rem_mask_vlen;
    return vreg_dst;
}

void jit_pp_ker_t::append_zp_src_comp(size_t offset, int idx, bool apply_mask) {
    const auto vreg_dst_masked_ = get_masked_vreg_dst(idx, apply_mask);
    const auto zp_src_comp_offset = offset * sizeof(int32_t);
    const auto zp_src_offset = jcp_.zp.src_is_common ? 0 : zp_src_comp_offset;
    const auto zp_src_comp_addr = ptr[reg_zp_src_comp_ + zp_src_comp_offset];
    const auto vreg_zp_src_comp = vreg_zp_comp_src(idx);
    const auto vreg_zp_src_comp_masked = vreg_zp_src_comp
            | (apply_mask ? kreg_rem_mask_short : kreg_rem_mask_vlen);

    vmovups(vreg_zp_src_comp_masked, zp_src_comp_addr);
    vpmulld(vreg_zp_src_comp_masked, vreg_zp_src_comp,
            EVEX_compress_addr(
                    reg_zp_src_, zp_src_offset, jcp_.zp.src_is_common));
    vcvtdq2ps(vreg_zp_src_comp, vreg_zp_src_comp);
    vaddps(vreg_dst_masked_, vreg_dst(idx), vreg_zp_src_comp);
}

void jit_pp_ker_t::apply_postops(const Xbyak::Reg64 &reg_dst, const int idx) {
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    if (jcp_.with_eltwise || jcp_.with_binary) {
        if (jcp_.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            const auto dst_offset_reg = reg_dst;
            const auto vmm_idx = vreg_dst_idx(idx);
            rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                    vmm_idx, ptr[abi_param1 + PARAM_OFF(g_oc_offset)]);
            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                    vmm_idx, reg_g_oc_off);
            rhs_arg_params.vmm_idx_to_out_off_oprnd.emplace(
                    vmm_idx, dst_offset_reg);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, dst_l_offset);
            rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);

            const injector_utils::register_preserve_guard_t register_guard(
                    this, {dst_offset_reg});
            sub(dst_offset_reg, ptr[abi_param1 + PARAM_OFF(dst_orig)]);
            const auto size = sizeof(jcp_.dst_data_type);
            if (size) shr(dst_offset_reg, std::log2(size));

            postops_injector_->compute_vector(
                    vreg_dst_idx(idx), rhs_arg_params);
        } else
            postops_injector_->compute_vector(vreg_dst_idx(idx));
    }
#undef PARAM_OFF
}

void jit_pp_ker_t::generate() {
    using namespace Xbyak;
    using namespace utils;

    // TODO: clean-up
    Reg64 reg_param = abi_param1;
    Reg64 reg_dst = rdx;
    Reg64 reg_acc = rax;
    Reg64 reg_bias = rbx;
    Reg64 reg_scales = rsi;

    Reg64 reg_len = r8;
    Reg64 reg_rem_mask_short = r10;
    Reg64 reg_rem_mask_vlen = r11;

    size_t vlen = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    for (; vlen >= 1 && (jcp_.oc % vlen != 0); --vlen) {}

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_scale = Zmm(1);
    Zmm vreg_nslope = Zmm(2);
    Zmm vreg_sum_scale = Zmm(3);
    Zmm vreg_signed_scale = Zmm(4);
    Zmm vreg_saturation_ubound = Zmm(5);

    if (jcp_.with_sum) {
        max_unroll = 8;
        zmm_step = 3;
    }

    preamble();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);

    if (jcp_.zp.src_exists) {
        mov(reg_zp_src_, ptr[reg_param + PARAM_OFF(zp_src)]);
        mov(reg_zp_src_comp_, ptr[reg_param + PARAM_OFF(zp_src_comp)]);
    }

    if (jcp_.zp.dst_exists)
        mov(reg_zp_dst_, ptr[reg_param + PARAM_OFF(zp_dst)]);

    vbroadcastss(vreg_nslope, ptr[reg_param + PARAM_OFF(nslope)]);
    vbroadcastss(vreg_sum_scale, ptr[reg_param + PARAM_OFF(sum_scale)]);
    vbroadcastss(vreg_signed_scale, ptr[reg_param + PARAM_OFF(signed_scale)]);
    if (jcp_.scale_idx_mult == 0) vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    mov(reg_rem_mask_vlen, 1);
    shl(reg_rem_mask_vlen, vlen);
    sub(reg_rem_mask_vlen, 1);
    kmovq(kreg_rem_mask_vlen, reg_rem_mask_vlen);

    if (jcp_.with_eltwise) vxorps(vreg_zero, vreg_zero, vreg_zero);
    init_saturate_f32(vreg_zero, vreg_saturation_ubound, reg_tmp_comp,
            data_type::f32, jcp_.dst_data_type);

    if (jcp_.with_binary) set_binary_postops_off(reg_oc_offset);

    // Load accumulated value, convert to float, apply sum (if any),
    // bias (if any), scaling, and relu (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        const auto &mask_reg
                = apply_mask ? kreg_rem_mask_short : kreg_rem_mask_vlen;

        if (jcp_.scale_idx_mult > 0) {
            assert(jcp_.scale_idx_mult == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            vreg_scale_ = vreg_scale_ | mask_reg;
            vmovups(vreg_scale_, scale_addr);
        }

        if (jcp_.with_binary) {
            if (offset) {
                advance_binary_postops_off(vlen);
                dst_l_offset += offset;
            }
            kmovq(opmask_binary, mask_reg);
        }
        const auto vreg_dst_ = get_masked_vreg_dst(idx, apply_mask);
        vcvtdq2ps(vreg_dst_, acc_addr);

        if (jcp_.zp.src_exists) append_zp_src_comp(offset, idx, apply_mask);

        if (jcp_.signed_input)
            vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (jcp_.with_bias) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            vreg_bias_ = vreg_bias_ | mask_reg;

            switch (jcp_.bias_data_type) {
                case data_type::s8: vpmovsxbd(vreg_bias_, bias_addr); break;
                case data_type::u8: vpmovzxbd(vreg_bias_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: vmovups(vreg_bias_, bias_addr); break;
                default: assert(!"unimplemented");
            }
            if (jcp_.bias_data_type != data_type::f32)
                vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];

        if (jcp_.with_sum) {
            auto vreg_prev_dst_ = vreg_prev_dst(idx);
            vreg_prev_dst_ = vreg_prev_dst_ | mask_reg;

            switch (jcp_.dst_data_type) {
                case data_type::f32:
                case data_type::s32: vmovups(vreg_prev_dst_, dst_addr); break;
                case data_type::s8: vpmovsxbd(vreg_prev_dst_, dst_addr); break;
                case data_type::u8: vpmovzxbd(vreg_prev_dst_, dst_addr); break;
                default: assert(!"unsupported data type");
            }
            if (jcp_.dst_data_type != data_type::f32)
                vcvtdq2ps(vreg_prev_dst(idx), vreg_prev_dst(idx));

            vfmadd231ps(vreg_dst(idx), vreg_prev_dst(idx), vreg_sum_scale);
        }

        apply_postops(reg_dst, idx);

        if (jcp_.zp.dst_exists) {
            const auto vreg_zp_dst_ = vreg_bias(idx);
            vbroadcastss(vreg_zp_dst_, ptr[reg_zp_dst_]);
            vcvtdq2ps(vreg_zp_dst_, vreg_zp_dst_);
            vaddps(vreg_dst_, vreg_dst_, vreg_zp_dst_);
        }

        if (one_of(jcp_.dst_data_type, data_type::u8, data_type::s8,
                    data_type::s32)) {
            saturate_f32(vreg_dst(idx), vreg_zero, vreg_saturation_ubound,
                    jcp_.dst_data_type);
            vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
        }

        switch (jcp_.dst_data_type) {
            case data_type::s8: vpmovsdb(dst_addr, vreg_dst_); break;
            case data_type::u8: vpmovusdb(dst_addr, vreg_dst_); break;
            case data_type::f32:
            case data_type::s32: vmovups(dst_addr, vreg_dst_); break;
            default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm
            = [&](const size_t offset, const size_t binary_offset) {
                  add(reg_dst, offset * dst_data_type_size_);
                  add(reg_acc, offset * sizeof(acc_data_t));
                  if (jcp_.with_binary) {
                      advance_binary_postops_off(binary_offset);
                  }
                  if (jcp_.scale_idx_mult) {
                      assert(jcp_.scale_idx_mult == 1);
                      add(reg_scales, offset * sizeof(float));
                  }
                  if (jcp_.with_bias)
                      add(reg_bias, offset * bias_data_type_size_);
                  if (jcp_.zp.src_exists) {
                      add(reg_zp_src_comp_, offset * sizeof(int32_t));
                      if (!jcp_.zp.src_is_common)
                          add(reg_zp_src_, offset * sizeof(int32_t));
                  }
              };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](const Reg64 offset, const Reg64 binary_offset) {
        lea(reg_dst, ptr[reg_dst + offset * dst_data_type_size_]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (jcp_.with_binary) { advance_binary_postops_off(binary_offset); }
        if (jcp_.scale_idx_mult) {
            assert(jcp_.scale_idx_mult == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (jcp_.with_bias)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);

        if (jcp_.zp.src_exists) {
            lea(reg_zp_src_comp_,
                    ptr[reg_zp_src_comp_ + offset * sizeof(int32_t)]);
            if (!jcp_.zp.src_is_common) {
                lea(reg_zp_src_, ptr[reg_zp_src_ + offset * sizeof(int32_t)]);
            }
        }
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (jcp_.with_bias) sub(reg_bias, jcp_.oc * bias_data_type_size_);
        if (jcp_.with_binary) {
            zero_binary_postops_off();
            dst_l_offset = 0;
        }
        if (jcp_.zp.src_exists) {
            const auto offset = jcp_.oc * sizeof(int32_t);
            sub(reg_zp_src_comp_, offset);
            if (!jcp_.zp.src_is_common) { sub(reg_zp_src_, offset); }
        }
        if (jcp_.scale_idx_mult) {
            assert(jcp_.scale_idx_mult == 1);
            sub(reg_scales, jcp_.oc * sizeof(float));
        }
        add(reg_dst, (jcp_.dst_os_stride - jcp_.oc) * dst_data_type_size_);
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    Label prologue_end;
    cmp(reg_oc_offset, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp, jcp_.oc);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp, vlen);
        jle(prologue_loop_tail, T_NEAR);
        L(prologue_loop);
        {
            compute(0, 0, false);
            advance_ptrs_imm(vlen, vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        mov(reg_rem_mask_short, 1);
        // cl == reg_tmp because reg_tmp <= vlen here
        shl(reg_rem_mask_short, cl);
        sub(reg_rem_mask_short, 1);
        jz(prologue_loop_end, T_NEAR);

        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp, reg_tmp);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, jcp_.oc);
        jle(main_loop_end, T_NEAR);

        Label main_loop;
        L(main_loop);
        {
            size_t OC_loop, OC_tail;
            if (static_cast<size_t>(jcp_.oc) < max_unroll * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = jcp_.oc;
            } else {
                OC_loop = vlen * def_unroll;
                OC_tail = jcp_.oc % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            const int vlen_tail = OC_tail % vlen;
            if (vlen_tail) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
            }

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(jcp_.oc, OC_loop));
                Label oc_loop;
                L(oc_loop);
                {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop, vlen);
                    sub(reg_tmp, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                const size_t oc_tail_rem = OC_tail % vlen;
                const size_t binary_offset = oc_tail_rem ? oc_tail_rem : vlen;
                advance_ptrs_imm(OC_tail, binary_offset);
            }

            rewind_ptrs();
            sub(reg_len, jcp_.oc);
            cmp(reg_len, jcp_.oc);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        cmp(reg_len, vlen);
        jle(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop);
        {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen, vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        mov(reg_rem_mask_short, 1);
        shl(reg_rem_mask_short, cl); // reg_tmp == rcx and reg_tail < vlen
        sub(reg_rem_mask_short, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    if (jcp_.with_eltwise) postops_injector_->prepare_table();
}

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
    if (!mayiuse(avx512_core)) return nullptr;
    return new jit_pp_ker_t(pd, jcp);
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
    using namespace x64::injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = false;
    return mayiuse(avx512_core)
            && dnnl::impl::cpu::x64::injector::post_ops_ok(
                    {avx512_core, {binary, eltwise, sum}, post_ops, dst_d,
                            sum_at_pos_0_only, sum_requires_scale_one});
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
