"""Deep profiling: per-phase op breakdown + backward analysis for MIMO training."""
import os, sys, time, re
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function

if not dist.is_initialized():
    dist.init_process_group(backend='nccl')
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
rank = dist.get_rank()

from benchmarks.mimo_throughput.config_loader import load_config
from benchmarks.mimo_throughput.training import create_mimo_model, forward_step, ENCODER_NAME
from benchmarks.mimo_throughput.data import SyntheticVLMIterator
from benchmarks.mimo_throughput.process_groups import ProcessGroupManager
from megatron.core.pipeline_parallel import schedules
from functools import partial

config = load_config("benchmarks/mimo_throughput/configs/1node_1b_7b/hetero_enc_tp2dp4_llm_tp4dp2.yaml")
config.validate()

pg_manager = ProcessGroupManager()
mimo_model, ctx = create_mimo_model(config, pg_manager)
encoder_grid, llm_grid, llm_pg = ctx['encoder_grid'], ctx['llm_grid'], ctx['llm_pg']

data_iter = SyntheticVLMIterator(
    encoder_hidden_size=config.encoder_arch.hidden_size,
    image_seq_length=config.encoder_arch.seq_length,
    total_seq_length=config.llm_arch.seq_length,
    micro_batch_size=config.data.micro_batch_size,
    vocab_size=config.llm_arch.vocab_size,
    image_token_id=config.data.image_token_id,
    encoder_name=ENCODER_NAME,
)

# Warmup
for _ in range(3):
    schedules.forward_backward_no_pipelining(
        forward_step_func=partial(forward_step, encoder_grid=encoder_grid, llm_grid=llm_grid, encoder_name=ENCODER_NAME),
        data_iterator=data_iter, model=[mimo_model],
        num_microbatches=config.data.num_microbatches,
        seq_length=config.llm_arch.seq_length, micro_batch_size=config.data.micro_batch_size,
        forward_only=False, pg_collection=llm_pg,
    )
torch.cuda.synchronize()

# Profile 1 iteration with full detail
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
) as prof:
    schedules.forward_backward_no_pipelining(
        forward_step_func=partial(forward_step, encoder_grid=encoder_grid, llm_grid=llm_grid, encoder_name=ENCODER_NAME),
        data_iterator=data_iter, model=[mimo_model],
        num_microbatches=config.data.num_microbatches,
        seq_length=config.llm_arch.seq_length, micro_batch_size=config.data.micro_batch_size,
        forward_only=False, pg_collection=llm_pg,
    )

if rank == 0:
    events = prof.events()

    # Classify ops
    def classify(name):
        n = name.lower()
        if any(k in n for k in ['nccl', 'allreduce', 'all_reduce', 'allgather', 'all_gather', 'reduce_scatter', 'broadcast', 'c10d::', 'record_param_comms']):
            return 'Communication'
        if any(k in n for k in ['gemm', 'nvjet', 'cutlass', 'cublas']):
            return 'GEMM'
        if any(k in n for k in ['fused_attn', 'fusedattn', 'sdpa', 'flash']):
            return 'Attention'
        if any(k in n for k in ['layernorm', 'ln_bwd', 'ln_fwd']):
            return 'LayerNorm'
        if any(k in n for k in ['dropout']):
            return 'Dropout'
        if any(k in n for k in ['gelu']):
            return 'Activation'
        if any(k in n for k in ['aten::add', 'aten::mul', 'aten::copy', 'elementwise']):
            return 'Elementwise'
        return 'Other'

    # ── Per-phase analysis using event tree ──
    # Find mimo:: events and their children
    mimo_events = [e for e in events if e.name.startswith("mimo::")]

    print("=" * 80)
    print("DEEP PROFILE ANALYSIS — PER-PHASE BREAKDOWN")
    print("=" * 80)
    print(f"Config: hetero enc TP2/DP4, llm TP4/DP2, mbs=2, nmb=2")
    print(f"Profiled: 1 iteration (2 microbatches fwd + bwd)")
    print()

    # Group mimo events by phase
    phases = {}
    for e in mimo_events:
        if e.name not in phases:
            phases[e.name] = {"total_cuda_us": 0, "count": 0, "children": []}
        phases[e.name]["total_cuda_us"] += e.cuda_time_total
        phases[e.name]["count"] += 1

        # Collect immediate CPU children (for understanding what's inside)
        if hasattr(e, 'cpu_children'):
            for child in e.cpu_children:
                phases[e.name]["children"].append(child)

    phase_order = [
        "mimo::text_embedding", "mimo::encoder_forward",
        "mimo::bridge_communicate", "mimo::align_embeddings",
        "mimo::llm_forward", "mimo::optimizer_step"
    ]

    total_phase_us = sum(p["total_cuda_us"] for p in phases.values())

    for phase_name in phase_order:
        if phase_name not in phases:
            continue
        info = phases[phase_name]
        short = phase_name.replace("mimo::", "")
        ms = info["total_cuda_us"] / 1000.0
        pct = 100.0 * info["total_cuda_us"] / max(total_phase_us, 1)

        print(f"{'─' * 80}")
        print(f"  {short} — {ms:.1f}ms ({pct:.1f}% of total, {info['count']} calls)")
        print(f"{'─' * 80}")

        # Analyze children by category
        child_categories = {}
        for child in info["children"]:
            cat = classify(child.name)
            if cat not in child_categories:
                child_categories[cat] = {"cuda_us": 0, "count": 0, "ops": {}}
            child_categories[cat]["cuda_us"] += child.cuda_time_total
            child_categories[cat]["count"] += 1
            # Track individual ops
            op_name = child.name[:60]
            if op_name not in child_categories[cat]["ops"]:
                child_categories[cat]["ops"][op_name] = {"cuda_us": 0, "count": 0}
            child_categories[cat]["ops"][op_name]["cuda_us"] += child.cuda_time_total
            child_categories[cat]["ops"][op_name]["count"] += 1

        if child_categories:
            for cat in sorted(child_categories, key=lambda c: -child_categories[c]["cuda_us"]):
                cinfo = child_categories[cat]
                cat_ms = cinfo["cuda_us"] / 1000.0
                cat_pct = 100.0 * cinfo["cuda_us"] / max(info["total_cuda_us"], 1)
                print(f"  {cat:<20} {cat_ms:>8.1f}ms  {cat_pct:>5.1f}%  ({cinfo['count']} ops)")

                # Top 3 ops in this category
                top_ops = sorted(cinfo["ops"].items(), key=lambda x: -x[1]["cuda_us"])[:3]
                for op_name, op_info in top_ops:
                    op_ms = op_info["cuda_us"] / 1000.0
                    print(f"    └─ {op_name[:55]:<55} {op_ms:>7.1f}ms ({op_info['count']}x)")
        else:
            print(f"  (no child events captured)")

        # Memory delta
        mem_bytes = getattr(mimo_model, 'memory_waterfall', {}).get(phase_name, 0)
        if mem_bytes:
            print(f"  Memory after: {mem_bytes / 1e9:.2f} GB")
        print()

    # ── Backward analysis ──
    # The backward is inside forward_backward_no_pipelining, not annotated with mimo::
    # Look for autograd::engine events
    backward_events = [e for e in events if 'backward' in e.name.lower() or 'autograd::engine' in e.name.lower()]
    bwd_cuda_total = sum(e.cuda_time_total for e in backward_events if 'evaluate_function' in e.name)

    print(f"{'─' * 80}")
    print(f"  BACKWARD PASS (autograd events)")
    print(f"{'─' * 80}")

    # Categorize backward ops
    bwd_categories = {}
    for e in events:
        if 'Backward' in e.name or 'backward' in e.name:
            cat = classify(e.name)
            if cat not in bwd_categories:
                bwd_categories[cat] = {"cuda_us": 0, "count": 0}
            bwd_categories[cat]["cuda_us"] += e.cuda_time_total
            bwd_categories[cat]["count"] += 1

    for cat in sorted(bwd_categories, key=lambda c: -bwd_categories[c]["cuda_us"]):
        cinfo = bwd_categories[cat]
        cat_ms = cinfo["cuda_us"] / 1000.0
        print(f"  {cat:<20} {cat_ms:>8.1f}ms  ({cinfo['count']} ops)")
    print()

    # ── Communication deep dive ──
    print(f"{'─' * 80}")
    print(f"  COMMUNICATION DEEP DIVE")
    print(f"{'─' * 80}")
    comm_events = [e for e in events if 'nccl' in e.name.lower() or 'c10d' in e.name.lower() or 'record_param_comms' in e.name.lower()]
    comm_by_type = {}
    for e in comm_events:
        key = e.name[:50]
        if key not in comm_by_type:
            comm_by_type[key] = {"cuda_us": 0, "count": 0}
        comm_by_type[key]["cuda_us"] += e.cuda_time_total
        comm_by_type[key]["count"] += 1

    for key in sorted(comm_by_type, key=lambda k: -comm_by_type[k]["cuda_us"])[:10]:
        info = comm_by_type[key]
        ms = info["cuda_us"] / 1000.0
        avg = ms / max(info["count"], 1)
        print(f"  {key[:50]:<50} {ms:>8.1f}ms  ({info['count']}x, avg {avg:.3f}ms)")
    print()

    # ── Wall clock vs CUDA time ──
    print(f"{'─' * 80}")
    print(f"  TIMING SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Phase CUDA sum:    {total_phase_us/1000:.1f}ms")
    print(f"  Peak GPU memory:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print()

del mimo_model
torch.cuda.empty_cache()
pg_manager.destroy_all()
dist.destroy_process_group()
