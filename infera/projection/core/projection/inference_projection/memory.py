###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference memory projection.

Per-rank HBM for serving is dominated by two terms that training does *not*
share:

  * **Weights only** — no gradients, no fp32 master copy, no Adam moments.
  * **KV cache** — grows with resident concurrency × context length.

plus a (comparatively small) **forward activation working set** for the
in-flight batch.  We also report how many concurrent sequences fit in the
remaining HBM, which is the headline capacity number for a serving config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

from infera.projection.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from infera.projection.core.projection.training_config import InferenceConfig, dtype_num_bytes

from .kv_cache import (
    KVCacheBreakdown,
    attention_dp_size,
    estimate_kv_cache,
    max_concurrent_sequences,
)


@dataclass
class InferenceMemoryResult:
    rank: int
    num_params: int
    weight_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    total_bytes: int
    kv: KVCacheBreakdown
    layers_on_rank: int
    hbm_capacity_bytes: Optional[int] = None
    max_concurrent_sequences: Optional[int] = None
    fits: Optional[bool] = None
    engine_reserved_bytes: int = 0


_GB = 1024.0 ** 3


def _layers_on_rank(inference_config: InferenceConfig) -> int:
    mc = inference_config.model_config
    pp = max(1, inference_config.model_parallel_config.pipeline_model_parallel_size)
    return max(1, (mc.num_layers + pp - 1) // pp)


def _forward_activation_bytes(inference_config: InferenceConfig) -> int:
    """Transient HBM for one serving scheduler step.

    Training ``estimated_activation_memory`` keeps SwiGLU gate+up+act for
    backward and an ``S×S`` score tensor. Fused/paged attention does not
    write scores to global memory, and inference does not store activations
    for backward. Leftover KV must not pay those.

    The working set is the engine's step token budget (vLLM
    ``--max-num-batched-tokens``), not resident ISL and not ``batch × seq``
    folded into one giant sequence.
    """
    req = inference_config.request_config
    batch = max(1, req.batch_size)
    tokens = batch * req.input_seq_len
    budget = int(req.max_num_batched_tokens or 0) or int(req.chunked_prefill_size or 0)
    if budget > 0:
        tokens = min(tokens, budget)

    mc = inference_config.model_config
    mp = inference_config.model_parallel_config
    tp = max(1, mp.tensor_model_parallel_size)
    cp = max(1, mp.context_model_parallel_size)
    tokens_per_rank = max(1, tokens // tp // cp)

    hidden = int(mc.hidden_size)
    ffn = int(mc.ffn_hidden_size or 0) or hidden
    if int(mc.num_experts or 0) > 0:
        ffn = int(mc.moe_ffn_hidden_size or 0) or ffn
        topk = max(1, int(mc.moe_router_topk or 1))
        ffn = ffn * topk
    # One live intermediate + hidden, bf16. Layer-at-a-time forward.
    return int(tokens_per_rank * (hidden + ffn) * 2)


def project_inference_memory(
    inference_config: InferenceConfig,
    *,
    rank: Optional[int] = None,
    hbm_capacity_gb: Optional[float] = None,
    verbose: bool = True,
) -> InferenceMemoryResult:
    eff_rank = int(os.getenv("RANK", "0")) if rank is None else int(rank)

    # A disaggregated deployment's memory ceiling is the decode pool's -- that is
    # where the KV cache lives and what caps concurrency. Everything below reads
    # parallelism off ``model_parallel_config``, so rebinding it once here points
    # the whole projection at the pool that actually holds the cache. Reporting
    # the global parallelism instead described a worker nobody runs, and once
    # per-pool TP existed without a matching global TP it reported a 744B model
    # as fitting zero sequences.
    #
    # The resident batch is split the same way the decode projector splits it:
    # each decode replica holds ``C / decode_replicas`` sequences, not the
    # system-wide concurrency. Sizing KV at the full batch on every replica
    # double-counted the cache by the replica count and rejected the 2-decode
    # MiniMax GB300 winner at 4096 in-flight -- 7 GB over 288 -- while silicon
    # ran it. Activations follow the same local batch.
    disagg = getattr(inference_config, "disaggregation_config", None)
    if disagg is not None and disagg.enabled:
        reps = max(1, int(disagg.decode_replicas or 1))
        local = max(1, int(inference_config.request_config.resolved_max_concurrency()) // reps)
        inference_config = replace(
            inference_config,
            model_parallel_config=disagg.decode_parallel(
                inference_config.model_parallel_config
            ),
            request_config=replace(
                inference_config.request_config,
                batch_size=local,
                max_concurrency=local,
            ),
        )

    view = inference_config.as_training_config(
        batch_size=inference_config.request_config.batch_size,
        seq_len=inference_config.request_config.input_seq_len,
    )
    # ``rank`` selects this rank's pipeline stage AND applies the profiler's
    # expert-parallel split, which is right for training, where EP is a separate
    # axis of GPUs from TP. A serving engine places experts on the same GPUs as
    # the tensor shards (vLLM sets EP = TP), so a rank holds 1/TP of every
    # weight class -- experts by expert assignment, everything else by tensor
    # sharding. Counting the expert split here as well would shard experts
    # twice and report a fraction of the weights a rank really loads.
    view.model_parallel_config = replace(
        view.model_parallel_config, expert_model_parallel_size=1
    )
    profiler = build_profiler(get_language_model_profiler_spec(view))

    # With experts left whole above, ``estimated_num_params`` counts this rank's
    # pipeline stage at full width and the TP divide below is the only split
    # applied. Norms and the router are replicated rather than sharded, and are
    # well under a percent of the total.
    tp = max(1, inference_config.model_parallel_config.tensor_model_parallel_size)
    num_params = profiler.estimated_num_params(rank=eff_rank) // tp
    weight_bytes = int(num_params * dtype_num_bytes(inference_config.request_config.weight_dtype))

    layers_on_rank = _layers_on_rank(inference_config)
    kv = estimate_kv_cache(inference_config, layers_on_rank)
    activation_bytes = _forward_activation_bytes(inference_config)

    total = weight_bytes + int(kv.bytes_total) + activation_bytes

    hbm_bytes = int(hbm_capacity_gb * _GB) if hbm_capacity_gb else None
    max_conc = None
    fits = None
    if hbm_bytes is not None:
        # Serving engines only hand a fraction of HBM to the runtime (vLLM
        # gpu_memory_utilization / SGLang mem_fraction_static). That fraction
        # is the *budget*; the engine still subtracts a profiled peak (weights
        # + dummy forward + HIP/allocator) from it. ``1 - fraction`` is extra
        # headroom outside the budget, not the CUDA context.
        fraction = inference_config.request_config.kv_cache_memory_fraction
        usable_bytes = int(hbm_bytes * float(fraction)) if fraction else hbm_bytes
        reserved = int(max(0.0, inference_config.request_config.engine_reserved_gb) * _GB)
        free_for_kv = usable_bytes - weight_bytes - activation_bytes - reserved
        # Blocks spilled to the host tier still hold a live session's context, so
        # they raise how many sessions a replica can keep resident even though
        # the actively-decoding batch stays in HBM. This is why the measured
        # agentic configs all run "dram" offload: agent sessions are idle most of
        # the time, and idle KV does not need HBM bandwidth.
        free_for_kv += inference_config.request_config.kv_offload_gb_per_gpu * _GB
        max_conc = max_concurrent_sequences(inference_config, layers_on_rank, free_for_kv)
        fits = (total + reserved) <= usable_bytes

    result = InferenceMemoryResult(
        rank=eff_rank,
        num_params=int(num_params),
        weight_bytes=weight_bytes,
        kv_cache_bytes=int(kv.bytes_total),
        activation_bytes=activation_bytes,
        total_bytes=total,
        kv=kv,
        layers_on_rank=layers_on_rank,
        hbm_capacity_bytes=hbm_bytes,
        max_concurrent_sequences=max_conc,
        fits=fits,
        engine_reserved_bytes=reserved if hbm_bytes is not None else 0,
    )

    if verbose:
        _print_memory(inference_config, result)
    return result


def _print_memory(inference_config: InferenceConfig, r: InferenceMemoryResult) -> None:
    req = inference_config.request_config
    print("\n" + "=" * 100)
    print(f"[inferasim:Inference] Memory Projection (Rank {r.rank})")
    print("=" * 100)
    print(f"  Params (this rank):       {r.num_params / 1e9:.4f} B")
    print(f"  Weights ({req.weight_dtype}):           {r.weight_bytes / _GB:.4f} GB")
    print(
        f"  KV cache ({req.kv_cache_dtype}):         {r.kv_cache_bytes / _GB:.4f} GB "
        f"(concurrency={r.kv.concurrency}, ctx={r.kv.max_context_len}, "
        f"layers/rank={r.layers_on_rank})"
    )
    print(f"    KV per sequence:        {r.kv.bytes_per_sequence / _GB:.4f} GB")
    attn_dp = attention_dp_size(inference_config)
    if attn_dp > 1:
        print(
            f"    Attention DP:           {attn_dp} "
            f"({r.kv.sequences_on_rank} sequences on this rank)"
        )
    print(f"  Activation working set:   {r.activation_bytes / _GB:.4f} GB")
    print(f"  Projected Total Memory:   {r.total_bytes / _GB:.4f} GB")
    if r.hbm_capacity_bytes is not None:
        print(f"  HBM capacity:             {r.hbm_capacity_bytes / _GB:.4f} GB")
        if req.kv_cache_memory_fraction:
            usable = r.hbm_capacity_bytes * float(req.kv_cache_memory_fraction)
            print(
                f"  Usable HBM (frac={req.kv_cache_memory_fraction:.2f}): {usable / _GB:.4f} GB"
            )
        if r.engine_reserved_bytes:
            print(f"  Engine reserved:          {r.engine_reserved_bytes / _GB:.4f} GB")
        print(f"  Fits:                     {r.fits}")
        if req.kv_offload_gb_per_gpu:
            print(
                f"  Host KV offload:          {req.kv_offload_gb_per_gpu:.1f} GB/GPU "
                f"@ {req.kv_offload_bw_gbps:.0f} GB/s"
            )
        print(f"  Max concurrent sequences: {r.max_concurrent_sequences}")
    print("=" * 100)
