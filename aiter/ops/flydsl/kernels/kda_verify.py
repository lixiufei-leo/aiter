# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 KDA target verify (speculative MTP) for gfx942 and gfx950.

One workgroup owns one (sequence, head) pair and walks the sequence's
``num_tokens`` draft tokens. Per token it runs the same chain as the fused KDA
decode in :mod:`.gdr_decode` -- the head-local ``f_b`` projection, the Q/K/V
causal convolution, the lower-bound decay gate, the gated delta rule and the
gated RMSNorm -- but with verify semantics:

* ``conv_state`` and ``state`` are read-only. Verify is speculative; the
  caller commits the accepted step afterwards.
* Each step's convolution window (the last three inputs) is written to
  ``intermediate_conv_window`` so the commit can roll the conv state forward.
* Each step's recurrent state is either written to ``intermediate_states``
  (snapshot mode), or the step's raw inputs -- pre-L2-norm K, pre-delta V, the
  log decay g and beta -- are written to the ReplaySSM rings (ring mode), which
  the commit replays.

The recurrent state stays in FP32 registers across the steps; a BF16 state is
widened on load and rounded only where a snapshot is stored.

The workgroup runs four phases separated by barriers:

1. The remaining threads convolve the head's Q, K and V channels (one thread
   per channel, keeping its 3-entry window in registers) while ``f_a`` is
   staged into LDS, widened to FP32 once for every thread that reads it.
2. The first 128 threads project ``f_a`` for every token and form the decay.
   Beside them, one wave L2-normalizes every token's Q and K and forms their
   dot product (eight lanes per token, in the recurrence's own lane layout and
   order), so the serial recurrence reads them from LDS instead of redoing
   them in every eight-lane group of every wave.
3. The recurrence. The eight-lane K sums are DPP butterflies, which leave the
   same bits in every lane of the group.
4. One wave per token applies the gated RMSNorm (each lane holds V rows l and
   l + 64, summed in the order of the two-wave reduction it replaces).

Every value is computed in the order of the fused KDA decode kernel, so the
first verify token still matches a decode step bit for bit.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .gdr_decode import (
    _KDA_DIM,
    _KDA_F32_VECTOR,
    _KDA_HEADS,
    _KDA_L2NORM_EPS,
    _KDA_PROJECTION_VECTOR,
    _KDA_SCALE,
    _KDA_STATE_DTYPES,
    _KDA_VECTOR_BYTES,
    _KDA_WARP_SIZE,
    _KDA_WARP_THREADS_K,
    _KDA_WARP_THREADS_V,
    _KDA_WAVE_SHUFFLE_OFFSETS,
    _kda_one_plus_exp_neg,
    _kda_sigmoid,
    _kda_vector_sum,
)
from .dpp_utils import dpp_xor_f32, update_dpp_i32
from .kernels_common import LOG2E as _LOG2E
from .tensor_shim import GTensor

# Draft tokens per sequence the kernel unrolls. Eight covers DSpark block
# sizes up to 7 plus the bonus token.
KDA_VERIFY_MAX_TOKENS = 8
# Waves per workgroup the kernel builds: 4 (32-row V groups) or 8 (64-row V
# groups, one thread per Q/K/V channel). The wrapper's automatic policy picks
# one of these from the batch, the ring or snapshot mode and the CU count.
KDA_VERIFY_NUM_WARPS = (4, 8)
# f_a is staged through LDS in 16-byte (8 x BF16) pieces, 16 per token.
_F_A_VECTOR = 8
_F_A_LANES = _KDA_DIM // _F_A_VECTOR
# The wave that normalizes Q and K beside the two gate waves.
_QK_NORM_WAVE = 2
# DPP control of the half-row mirror (lane i <-> 7 - i within eight lanes).
_DPP_ROW_HALF_MIRROR = 0x141


def _kda_k_group_sum_dpp(value):
    """Sum a value across the eight lanes that split one K row.

    The same butterfly as ``gdr_decode._kda_k_group_sum`` (xor 1, xor 2, then
    the other quad), done with DPP moves instead of ``ds_swizzle``: xor 1 and
    xor 2 by quad permutes, and the half-row mirror pairs each lane with one in
    the other quad. Each level adds the same two partials in every lane, so
    the result is bit-identical and the same in all eight lanes.
    """
    value = value + dpp_xor_f32(value, 1)
    value = value + dpp_xor_f32(value, 2)
    bits = fx.Float32(value).bitcast(fx.Int32)
    mirrored = update_dpp_i32(bits, bits, _DPP_ROW_HALF_MIRROR, 0xF, 0xF, False)
    return value + mirrored.bitcast(T.f32)


# Each entry holds a JIT launcher and its compiled module, which CUDA graphs
# captured earlier may still launch, so specializations are not evicted under
# normal use (sized like the KDA decode cache).
@functools.lru_cache(maxsize=1024)
def create_kda_verify_kernel(
    norm_eps: float,
    lower_bound: float,
    num_tokens: int,
    state_dtype: str,
    cache_ring: bool,
    num_warps: int = 4,
):
    """Build the Kimi-K3 KDA verify specialization (gfx942/gfx950).

    ``num_tokens`` is the fixed draft-token count per sequence (1 to
    :data:`KDA_VERIFY_MAX_TOKENS`). ``state_dtype`` is the recurrent-state
    storage, ``"f32"`` or ``"bf16"``; snapshots use the same storage.
    ``cache_ring`` selects ReplaySSM ring writes instead of state snapshots.
    ``num_warps`` (one of :data:`KDA_VERIFY_NUM_WARPS`, 4 or 8) splits the V
    rows of a head across more waves, which shortens each serial step at the
    cost of fewer resident workgroups.
    """
    if state_dtype not in _KDA_STATE_DTYPES:
        raise ValueError(
            f"unsupported KDA state dtype {state_dtype!r}; "
            f"expected one of {sorted(_KDA_STATE_DTYPES)}"
        )
    if not 1 <= num_tokens <= KDA_VERIFY_MAX_TOKENS:
        raise ValueError(
            f"num_tokens must be in [1, {KDA_VERIFY_MAX_TOKENS}], got {num_tokens}"
        )
    if num_warps not in KDA_VERIFY_NUM_WARPS:
        raise ValueError(
            f"num_warps must be one of {KDA_VERIFY_NUM_WARPS}, got {num_warps}"
        )
    block_threads = num_warps * _KDA_WARP_SIZE
    v_group_tile = num_warps * _KDA_WARP_THREADS_V
    v_iters = _KDA_DIM // v_group_tile
    # With 8 waves each Q/K/V channel gets its own thread; with 4 waves one
    # thread convolves the head's Q, K and V channel.
    split_conv = block_threads >= 4 * _KDA_DIM
    f32_state = state_dtype == "f32"
    state_num = _KDA_STATE_DTYPES[state_dtype]
    state_bytes = state_num.width // 8
    values_per_thread_k = _KDA_VECTOR_BYTES // state_bytes
    warp_tile_k = _KDA_WARP_THREADS_K * values_per_thread_k
    k_iters = _KDA_DIM // warp_tile_k
    num_steps = num_tokens
    token_rows = num_steps * _KDA_DIM
    # f32 LDS arrays are padded to a 16-byte multiple.
    beta_rows = max(num_steps, _KDA_F32_VECTOR)
    name_suffix = (
        f"_t{num_steps}"
        + ("" if f32_state else f"_{state_dtype}state")
        + ("_ring" if cache_ring else "")
        + ("" if num_warps == 4 else f"_w{num_warps}")
    )

    @fx.struct
    class SharedStorage:
        f_a: fx.Array[fx.Float32, token_rows, _KDA_VECTOR_BYTES]
        q: fx.Array[fx.BFloat16, token_rows, _KDA_VECTOR_BYTES]
        k: fx.Array[fx.BFloat16, token_rows, _KDA_VECTOR_BYTES]
        v: fx.Array[fx.BFloat16, token_rows, _KDA_VECTOR_BYTES]
        q_hat: fx.Array[fx.Float32, token_rows, _KDA_VECTOR_BYTES]
        k_hat: fx.Array[fx.Float32, token_rows, _KDA_VECTOR_BYTES]
        decay: fx.Array[fx.Float32, token_rows, _KDA_VECTOR_BYTES]
        beta: fx.Array[fx.Float32, beta_rows, _KDA_VECTOR_BYTES]
        dot_kq: fx.Array[fx.Float32, beta_rows, _KDA_VECTOR_BYTES]
        recurrent_out: fx.Array[fx.BFloat16, token_rows, _KDA_VECTOR_BYTES]

    @flyc.kernel(
        name=f"kda_verify_fused_projection_bf16{name_suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        f_a_mem: fx.Tensor,
        f_b_weight_mem: fx.Tensor,
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        cache_indices_mem: fx.Tensor,
        inter_indices_mem: fx.Tensor,
        window_mem: fx.Tensor,
        inter_state_mem: fx.Tensor,
        ring_v_mem: fx.Tensor,
        ring_k_mem: fx.Tensor,
        ring_g_mem: fx.Tensor,
        ring_beta_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        num_cache_slots: fx.Int32,
        num_inter_lines: fx.Int32,
        stride_f_a_token: fx.Int32,
        stride_f_b_head: fx.Int32,
        stride_f_b_output: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_window_line: fx.Int64,
        stride_window_step: fx.Int32,
        stride_window_channel: fx.Int32,
        stride_window_width: fx.Int32,
        stride_inter_line: fx.Int64,
        stride_inter_step: fx.Int32,
        stride_ring_v_slot: fx.Int64,
        stride_ring_v_head: fx.Int32,
        stride_ring_v_token: fx.Int32,
        stride_ring_k_slot: fx.Int64,
        stride_ring_k_head: fx.Int32,
        stride_ring_k_token: fx.Int32,
        stride_ring_g_slot: fx.Int64,
        stride_ring_g_head: fx.Int32,
        stride_ring_g_token: fx.Int32,
        stride_ring_beta_slot: fx.Int64,
        stride_ring_beta_head: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
    ):
        f_a = GTensor(f_a_mem, dtype=T.bf16, shape=(-1,))
        f_b_weight = GTensor(f_b_weight_mem, dtype=T.bf16, shape=(-1,))
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        weight = GTensor(weight_mem, dtype=T.f32, shape=(-1,))
        raw_beta = GTensor(raw_beta_mem, dtype=T.bf16, shape=(-1,))
        A_log = GTensor(A_log_mem, dtype=T.f32, shape=(-1,))
        dt_bias = GTensor(dt_bias_mem, dtype=T.f32, shape=(-1,))
        cache_indices = GTensor(cache_indices_mem, dtype=T.i32, shape=(-1,))
        inter_indices = GTensor(inter_indices_mem, dtype=T.i32, shape=(-1,))
        output_gate = GTensor(output_gate_mem, dtype=T.bf16, shape=(-1,))
        norm_weight = GTensor(norm_weight_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        f_a_lds = shared.f_a.ptr
        q_lds = shared.q.ptr
        k_lds = shared.k.ptr
        v_lds = shared.v.ptr
        q_hat_lds = shared.q_hat.ptr
        k_hat_lds = shared.k_hat.ptr
        decay_lds = shared.decay.ptr
        beta_lds = shared.beta.ptr
        dot_lds = shared.dot_kq.ptr
        out_lds = shared.recurrent_out.ptr

        tid = fx.thread_idx.x
        block = fx.block_idx.x
        batch = block // _KDA_HEADS
        head = block % _KDA_HEADS
        lane = tid % _KDA_WARP_SIZE
        warp = tid // _KDA_WARP_SIZE
        lane_k = lane % _KDA_WARP_THREADS_K
        token0 = batch * num_steps

        cache_idx = cache_indices[batch]
        inter_idx = inter_indices[batch]
        # Negative entries (PAD_SLOT_ID) are padding; slot zero is live.
        # Indices past either cache are treated as padding instead of being
        # dereferenced out of bounds.
        valid = (
            (cache_idx >= 0)
            & (cache_idx < num_cache_slots)
            & (inter_idx >= 0)
            & (inter_idx < num_inter_lines)
        )
        # Rebase every slot-indexed cache in 64-bit; buffer accesses take a
        # 32-bit byte offset.
        cache_slot = cache_idx.to(fx.Int64)
        inter_line = inter_idx.to(fx.Int64)
        conv_state = GTensor(
            conv_state_mem,
            dtype=T.bf16,
            shape=(-1,),
            static_bytes_offset_i64=cache_slot
            * stride_conv_slot
            * fx.Int64(fx.BFloat16.width // 8),
        )
        state = GTensor(
            state_mem,
            dtype=state_num.ir_type,
            shape=(-1,),
            static_bytes_offset_i64=cache_slot
            * stride_state_slot
            * fx.Int64(state_bytes),
        )
        window = GTensor(
            window_mem,
            dtype=T.bf16,
            shape=(-1,),
            static_bytes_offset_i64=inter_line
            * stride_window_line
            * fx.Int64(fx.BFloat16.width // 8),
        )
        if const_expr(cache_ring):
            ring_v = GTensor(
                ring_v_mem,
                dtype=T.bf16,
                shape=(-1,),
                static_bytes_offset_i64=cache_slot
                * stride_ring_v_slot
                * fx.Int64(fx.BFloat16.width // 8),
            )
            ring_k = GTensor(
                ring_k_mem,
                dtype=T.bf16,
                shape=(-1,),
                static_bytes_offset_i64=cache_slot
                * stride_ring_k_slot
                * fx.Int64(fx.BFloat16.width // 8),
            )
            ring_g = GTensor(
                ring_g_mem,
                dtype=T.f32,
                shape=(-1,),
                static_bytes_offset_i64=cache_slot
                * stride_ring_g_slot
                * fx.Int64(fx.Float32.width // 8),
            )
            ring_beta = GTensor(
                ring_beta_mem,
                dtype=T.f32,
                shape=(-1,),
                static_bytes_offset_i64=cache_slot
                * stride_ring_beta_slot
                * fx.Int64(fx.Float32.width // 8),
            )
        else:
            inter_state = GTensor(
                inter_state_mem,
                dtype=state_num.ir_type,
                shape=(-1,),
                static_bytes_offset_i64=inter_line
                * stride_inter_line
                * fx.Int64(state_bytes),
            )

        # The runtime-if rewriter carries every object whose method is called
        # in a branch, which tensor views cannot be. Each phase is therefore a
        # closure; the dispatch at the end calls them under the runtime guards.
        def stage_f_a():
            # Threads 0..16*num_steps-1 copy 16 bytes of one token's f_a row,
            # widened to FP32 here once: every gate thread reads every value,
            # and the widening is exact.
            step = tid // _F_A_LANES
            offset = tid % _F_A_LANES * _F_A_VECTOR
            values = f_a.vec_load(
                ((token0 + step) * stride_f_a_token + offset,), _F_A_VECTOR
            ).to(fx.Float32)
            row = f_a_lds + step * _KDA_DIM + offset
            fx.ptr_store(values.shuffle(values, [0, 1, 2, 3]), row)
            fx.ptr_store(values.shuffle(values, [4, 5, 6, 7]), row + _KDA_F32_VECTOR)

        def prepare_gates():
            # Threads 0..127 own one K channel of the head. The f_b weight row
            # is read once in 16-byte loads and applied to every token. The
            # FP32 accumulation order is the decode kernel's (four lanes over
            # 32 slices), so a verify token matches a decode step bit for bit.
            channel = tid
            accums = [
                fx.Vector.filled(_KDA_PROJECTION_VECTOR, 0.0, fx.Float32)
                for _ in range(num_steps)
            ]
            f_b_base = head * stride_f_b_head + channel * stride_f_b_output
            vec_f32 = fx.Vector[fx.Float32, _KDA_F32_VECTOR]
            lo = list(range(_KDA_PROJECTION_VECTOR))
            hi = list(range(_KDA_PROJECTION_VECTOR, _F_A_VECTOR))
            # Issue the whole f_b row before the first use, so its loads share
            # one memory round trip instead of a few in flight at a time.
            rows = [
                f_b_weight.vec_load((f_b_base + i * _F_A_VECTOR,), _F_A_VECTOR)
                for i in range_constexpr(_KDA_DIM // _F_A_VECTOR)
            ]
            for i in range_constexpr(_KDA_DIM // _F_A_VECTOR):
                w = rows[i].to(fx.Float32)
                for step in range_constexpr(num_steps):
                    f_a_row = f_a_lds + step * _KDA_DIM + i * _F_A_VECTOR
                    f_a_lo = fx.ptr_load(f_a_row, vec_f32)
                    f_a_hi = fx.ptr_load(f_a_row + _KDA_F32_VECTOR, vec_f32)
                    f_a_values = f_a_lo.shuffle(f_a_hi, list(range(_F_A_VECTOR)))
                    acc = fx.math.fma(
                        f_a_values.shuffle(f_a_values, lo),
                        w.shuffle(w, lo),
                        accums[step],
                    )
                    accums[step] = fx.math.fma(
                        f_a_values.shuffle(f_a_values, hi), w.shuffle(w, hi), acc
                    )
            a = fx.math.exp2(A_log[head] * fx.Float32(_LOG2E))
            dt = dt_bias[head * _KDA_DIM + channel]
            for step in range_constexpr(num_steps):
                projected = fx.BFloat16(_kda_vector_sum(accums[step]))
                log_decay = _kda_sigmoid((fx.Float32(projected) + dt) * a, lower_bound)
                fx.ptr_store(
                    fx.math.exp2(log_decay * fx.Float32(_LOG2E)),
                    decay_lds + step * _KDA_DIM + channel,
                )
                if const_expr(cache_ring):
                    ring_g.store(
                        head * stride_ring_g_head
                        + step * stride_ring_g_token
                        + channel,
                        log_decay,
                    )

        def convolve_channel(
            channel, local, lds, ring, stride_ring_head, stride_ring_token
        ):
            # One thread owns the channel for every step, so the 3-entry
            # window stays in registers. The cache itself is not advanced.
            cs_base = channel * stride_conv_channel
            c0 = conv_state[cs_base]
            c1 = conv_state[cs_base + stride_conv_width]
            c2 = conv_state[cs_base + 2 * stride_conv_width]
            w_base = channel * stride_weight_channel
            w0 = weight[w_base]
            w1 = weight[w_base + stride_weight_width]
            w2 = weight[w_base + 2 * stride_weight_width]
            w3 = weight[w_base + 3 * stride_weight_width]
            # Issue every token's load before the window stores: a buffer load
            # cannot move above a possibly aliasing store, so interleaving them
            # serializes one memory latency per step.
            inputs = [
                x[(token0 + step) * stride_x_token + channel]
                for step in range_constexpr(num_steps)
            ]
            for step in range_constexpr(num_steps):
                current = inputs[step]
                acc = fx.Float32(c0) * w0
                acc = acc + fx.Float32(c1) * w1
                acc = acc + fx.Float32(c2) * w2
                acc = acc + fx.Float32(current) * w3
                c0 = c1
                c1 = c2
                c2 = current
                window_base = (
                    step * stride_window_step + channel * stride_window_channel
                )
                window.store(window_base, c0)
                window.store(window_base + stride_window_width, c1)
                window.store(window_base + 2 * stride_window_width, c2)
                # SiLU as a single division matches the decode kernel's rounding.
                value = (acc / _kda_one_plus_exp_neg(acc)).to(fx.BFloat16)
                fx.ptr_store(value, lds + step * _KDA_DIM + local)
                if const_expr(ring is not None):
                    ring.store(
                        head * stride_ring_head + step * stride_ring_token + local,
                        value,
                    )

        def convolve_q(local):
            convolve_channel(head * _KDA_DIM + local, local, q_lds, None, 0, 0)

        def convolve_k(local):
            convolve_channel(
                (_KDA_HEADS + head) * _KDA_DIM + local,
                local,
                k_lds,
                ring_k if cache_ring else None,
                stride_ring_k_head,
                stride_ring_k_token,
            )

        def convolve_v(local):
            convolve_channel(
                (2 * _KDA_HEADS + head) * _KDA_DIM + local,
                local,
                v_lds,
                ring_v if cache_ring else None,
                stride_ring_v_head,
                stride_ring_v_token,
            )

        def prepare_qkv():
            # With 4 waves, threads 128..255 own one Q, K and V channel each.
            local = tid - _KDA_DIM
            convolve_q(local)
            convolve_k(local)
            convolve_v(local)

        def prepare_beta():
            # Threads 0..num_steps-1 own one token each.
            beta = _kda_sigmoid(
                fx.Float32(raw_beta[(token0 + tid) * stride_beta_token + head])
            )
            fx.ptr_store(beta, beta_lds + tid)
            if const_expr(cache_ring):
                ring_beta.store(head * stride_ring_beta_head + tid, beta)

        def load_f32_k(base, offset):
            # values_per_thread_k FP32 LDS values, in 16-byte pieces.
            if const_expr(values_per_thread_k == 2 * _KDA_F32_VECTOR):
                vec_f32 = fx.Vector[fx.Float32, _KDA_F32_VECTOR]
                lo = fx.ptr_load(base + offset, vec_f32)
                hi = fx.ptr_load(base + offset + _KDA_F32_VECTOR, vec_f32)
                return lo.shuffle(hi, list(range(values_per_thread_k)))
            return fx.ptr_load(base + offset, fx.Vector[fx.Float32, values_per_thread_k])

        def store_f32_k(base, offset, value):
            if const_expr(values_per_thread_k == 2 * _KDA_F32_VECTOR):
                fx.ptr_store(
                    value.shuffle(value, list(range(_KDA_F32_VECTOR))), base + offset
                )
                fx.ptr_store(
                    value.shuffle(
                        value, list(range(_KDA_F32_VECTOR, 2 * _KDA_F32_VECTOR))
                    ),
                    base + offset + _KDA_F32_VECTOR,
                )
            else:
                fx.ptr_store(value, base + offset)

        def load_decay(offset):
            return load_f32_k(decay_lds, offset)

        def normalize_qk():
            # One wave, eight lanes per token (lane_k owns the same K slices
            # as in the recurrence), so the sums run in the recurrence's -- and
            # the decode kernel's -- order. Groups past the last token repeat
            # it and store the same bits again.
            group = lane // _KDA_WARP_THREADS_K
            step = (group < num_steps).select(group, fx.Int32(num_steps - 1))
            row = step * _KDA_DIM
            k_vec_start = lane_k * values_per_thread_k
            vec_bf16 = fx.Vector[fx.BFloat16, values_per_thread_k]
            zero_vec = fx.Vector.filled(values_per_thread_k, 0.0, fx.Float32)
            q_vecs = []
            k_vecs = []
            sum_q_partial = fx.Float32(0.0)
            sum_k_partial = fx.Float32(0.0)
            for ki in range_constexpr(k_iters):
                k_base = k_vec_start + ki * warp_tile_k
                q_f32 = fx.ptr_load(q_lds + row + k_base, vec_bf16).to(fx.Float32)
                k_f32 = fx.ptr_load(k_lds + row + k_base, vec_bf16).to(fx.Float32)
                q_vecs.append(q_f32)
                k_vecs.append(k_f32)
                sum_q_partial = sum_q_partial + _kda_vector_sum(q_f32 * q_f32)
                sum_k_partial = sum_k_partial + _kda_vector_sum(k_f32 * k_f32)
            norm_q = _kda_k_group_sum_dpp(sum_q_partial)
            norm_k = _kda_k_group_sum_dpp(sum_k_partial)
            inv_q = fx.math.rsqrt(norm_q + fx.Float32(_KDA_L2NORM_EPS))
            inv_k = fx.math.rsqrt(norm_k + fx.Float32(_KDA_L2NORM_EPS))
            dot_kq_vec = zero_vec
            for ki in range_constexpr(k_iters):
                q_vecs[ki] = q_vecs[ki] * inv_q * fx.Float32(_KDA_SCALE)
                k_vecs[ki] = k_vecs[ki] * inv_k
                dot_kq_vec = fx.math.fma(k_vecs[ki], q_vecs[ki], dot_kq_vec)
                k_base = k_vec_start + ki * warp_tile_k
                store_f32_k(q_hat_lds, row + k_base, q_vecs[ki])
                store_f32_k(k_hat_lds, row + k_base, k_vecs[ki])
            dot_kq = _kda_k_group_sum_dpp(_kda_vector_sum(dot_kq_vec))
            if lane_k == 0:
                fx.ptr_store(dot_kq, dot_lds + step)

        def load_state():
            # Each wave owns 8 V rows of every V group (32 rows with 4 waves,
            # 64 with 8) and eight-lane subgroups reduce K, as in the decode
            # kernel. The loads are issued before the first phase, so their
            # memory latency overlaps the convolution and the gates.
            k_vec_start = lane_k * values_per_thread_k
            global_v_start = warp * _KDA_WARP_THREADS_V + lane // _KDA_WARP_THREADS_K
            state_head_base = head * (_KDA_DIM * _KDA_DIM)
            raw = []
            for vi in range_constexpr(v_iters):
                global_v = global_v_start + vi * v_group_tile
                for ki in range_constexpr(k_iters):
                    k_base = k_vec_start + ki * warp_tile_k
                    state_off = state_head_base + global_v * _KDA_DIM + k_base
                    raw.append(state.vec_load((state_off,), values_per_thread_k))
            return raw

        def recurrent_verify(raw_state):
            # The state stays in FP32 registers across the steps.
            k_vec_start = lane_k * values_per_thread_k
            global_v_start = warp * _KDA_WARP_THREADS_V + lane // _KDA_WARP_THREADS_K
            zero_vec = fx.Vector.filled(values_per_thread_k, 0.0, fx.Float32)
            state_head_base = head * (_KDA_DIM * _KDA_DIM)
            state_vecs = [vec.to(fx.Float32) for vec in raw_state]

            for step in range_constexpr(num_steps):
                row = step * _KDA_DIM
                # Q and K come normalized (and Q scaled) from normalize_qk.
                q_vecs = []
                k_vecs = []
                decay_vecs = []
                for ki in range_constexpr(k_iters):
                    k_base = k_vec_start + ki * warp_tile_k
                    q_vecs.append(load_f32_k(q_hat_lds, row + k_base))
                    k_vecs.append(load_f32_k(k_hat_lds, row + k_base))
                    decay_vecs.append(load_decay(row + k_base))
                dot_kq = fx.ptr_load(dot_lds + step)
                beta = fx.ptr_load(beta_lds + step)

                for vi in range_constexpr(v_iters):
                    global_v = global_v_start + vi * v_group_tile
                    sum_hk_vec = zero_vec
                    sum_hq_vec = zero_vec
                    for ki in range_constexpr(k_iters):
                        state_pos = vi * k_iters + ki
                        decayed = state_vecs[state_pos] * decay_vecs[ki]
                        state_vecs[state_pos] = decayed
                        sum_hk_vec = fx.math.fma(decayed, k_vecs[ki], sum_hk_vec)
                        sum_hq_vec = fx.math.fma(decayed, q_vecs[ki], sum_hq_vec)

                    sum_hk = _kda_k_group_sum_dpp(_kda_vector_sum(sum_hk_vec))
                    sum_hq = _kda_k_group_sum_dpp(_kda_vector_sum(sum_hq_vec))

                    # Every lane of the group holds the same sums, so v_new
                    # needs no broadcast from the group leader.
                    conv_v = fx.Float32(fx.ptr_load(v_lds + row + global_v))
                    v_new = (conv_v - sum_hk) * beta
                    recurrent_value = sum_hq + v_new * dot_kq
                    v_new_vec = fx.Vector.filled(values_per_thread_k, v_new, fx.Float32)

                    for ki in range_constexpr(k_iters):
                        state_pos = vi * k_iters + ki
                        updated = fx.math.fma(
                            k_vecs[ki], v_new_vec, state_vecs[state_pos]
                        )
                        state_vecs[state_pos] = updated
                        if const_expr(not cache_ring):
                            k_base = k_vec_start + ki * warp_tile_k
                            inter_state.vec_store(
                                (
                                    step * stride_inter_step
                                    + state_head_base
                                    + global_v * _KDA_DIM
                                    + k_base,
                                ),
                                updated.to(state_num),
                                values_per_thread_k,
                            )

                    if lane_k == 0:
                        fx.ptr_store(
                            fx.BFloat16(recurrent_value), out_lds + row + global_v
                        )

        def store_gated_outputs(step):
            # One wave per token. Lane l holds V rows l and l + 64 and sums
            # each half over the wave, then adds the halves: the order of the
            # two-wave reduction in the decode kernel. The recurrent values
            # keep the model's BF16 boundary before RMSNorm and gating.
            token = token0 + step
            gate_base = token * stride_gate_token + head * stride_gate_head
            out_base = token * stride_out_token + head * stride_out_head
            # Load the gates before the first output store: a buffer load
            # cannot move above a possibly aliasing store.
            channels = [lane + half * _KDA_WARP_SIZE for half in range(2)]
            norm_w = [fx.Float32(norm_weight[c]) for c in channels]
            gates = [fx.Float32(output_gate[gate_base + c]) for c in channels]
            recurrent = [
                fx.Float32(fx.ptr_load(out_lds + step * _KDA_DIM + c))
                for c in channels
            ]
            squares = [r * r for r in recurrent]
            for offset in _KDA_WAVE_SHUFFLE_OFFSETS:
                squares = [
                    sq + sq.shuffle_xor(offset, _KDA_WARP_SIZE) for sq in squares
                ]
            inv_rms = fx.math.rsqrt(
                (squares[0] + squares[1]) * fx.Float32(1.0 / _KDA_DIM)
                + fx.Float32(norm_eps)
            )
            for half in range_constexpr(2):
                result = (
                    recurrent[half] * inv_rms * norm_w[half] * _kda_sigmoid(gates[half])
                )
                out.store(out_base + channels[half], result.to(fx.BFloat16))

        def store_zero_outputs():
            for step in range_constexpr(num_steps):
                out.store(
                    (token0 + step) * stride_out_token + head * stride_out_head + tid,
                    fx.BFloat16(0.0),
                )

        if valid:
            raw_state = load_state()
            # The convolution and beta need no staged input, so they run
            # beside the f_a staging instead of after its barrier.
            if tid < num_steps * _F_A_LANES:
                stage_f_a()
            if const_expr(split_conv):
                if (tid >= _KDA_DIM) & (tid < 2 * _KDA_DIM):
                    convolve_q(tid - _KDA_DIM)
                if (tid >= 2 * _KDA_DIM) & (tid < 3 * _KDA_DIM):
                    convolve_k(tid - 2 * _KDA_DIM)
                if (tid >= 3 * _KDA_DIM) & (tid < 4 * _KDA_DIM):
                    convolve_v(tid - 3 * _KDA_DIM)
            else:
                if tid >= _KDA_DIM:
                    prepare_qkv()
            if tid < num_steps:
                prepare_beta()
            fx.gpu.barrier()
            # Waves 0 and 1 project the gates; wave 2 (on another SIMD)
            # normalizes Q and K for every token meanwhile.
            if tid < _KDA_DIM:
                prepare_gates()
            if warp == _QK_NORM_WAVE:
                normalize_qk()
            # Gate, normalized Q/K, V and beta LDS rows must be visible before
            # the recurrence.
            fx.gpu.barrier()
            recurrent_verify(raw_state)
            fx.gpu.barrier()
            for j in range_constexpr(-(-num_steps // num_warps)):
                step = warp + j * num_warps
                if step < num_steps:
                    store_gated_outputs(step)
        else:
            if tid < _KDA_DIM:
                store_zero_outputs()

    @flyc.jit
    def launch(
        f_a_mem: fx.Tensor,
        f_b_weight_mem: fx.Tensor,
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        cache_indices_mem: fx.Tensor,
        inter_indices_mem: fx.Tensor,
        window_mem: fx.Tensor,
        inter_state_mem: fx.Tensor,
        ring_v_mem: fx.Tensor,
        ring_k_mem: fx.Tensor,
        ring_g_mem: fx.Tensor,
        ring_beta_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        batch_size: fx.Int32,
        num_cache_slots: fx.Int32,
        num_inter_lines: fx.Int32,
        stride_f_a_token: fx.Int32,
        stride_f_b_head: fx.Int32,
        stride_f_b_output: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_window_line: fx.Int64,
        stride_window_step: fx.Int32,
        stride_window_channel: fx.Int32,
        stride_window_width: fx.Int32,
        stride_inter_line: fx.Int64,
        stride_inter_step: fx.Int32,
        stride_ring_v_slot: fx.Int64,
        stride_ring_v_head: fx.Int32,
        stride_ring_v_token: fx.Int32,
        stride_ring_k_slot: fx.Int64,
        stride_ring_k_head: fx.Int32,
        stride_ring_k_token: fx.Int32,
        stride_ring_g_slot: fx.Int64,
        stride_ring_g_head: fx.Int32,
        stride_ring_g_token: fx.Int32,
        stride_ring_beta_slot: fx.Int64,
        stride_ring_beta_head: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            f_a_mem,
            f_b_weight_mem,
            x_mem,
            weight_mem,
            conv_state_mem,
            raw_beta_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            cache_indices_mem,
            inter_indices_mem,
            window_mem,
            inter_state_mem,
            ring_v_mem,
            ring_k_mem,
            ring_g_mem,
            ring_beta_mem,
            output_gate_mem,
            norm_weight_mem,
            out_mem,
            num_cache_slots,
            num_inter_lines,
            stride_f_a_token,
            stride_f_b_head,
            stride_f_b_output,
            stride_x_token,
            stride_weight_channel,
            stride_weight_width,
            stride_conv_slot,
            stride_conv_channel,
            stride_conv_width,
            stride_beta_token,
            stride_state_slot,
            stride_window_line,
            stride_window_step,
            stride_window_channel,
            stride_window_width,
            stride_inter_line,
            stride_inter_step,
            stride_ring_v_slot,
            stride_ring_v_head,
            stride_ring_v_token,
            stride_ring_k_slot,
            stride_ring_k_head,
            stride_ring_k_token,
            stride_ring_g_slot,
            stride_ring_g_head,
            stride_ring_g_token,
            stride_ring_beta_slot,
            stride_ring_beta_head,
            stride_gate_token,
            stride_gate_head,
            stride_out_token,
            stride_out_head,
        ).launch(
            grid=(batch_size * _KDA_HEADS, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    # The projection and the per-step snapshot stores raise register
    # pressure, as in the fused-projection decode kernel.
    launch.compile_hints = {
        "waves_per_eu": 2,
        "llvm_options": {
            "amdgpu-expert-scheduling-mode": True,
        },
    }
    return launch
