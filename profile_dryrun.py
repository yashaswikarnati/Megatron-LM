"""Dry run: capture rich profiler data from one iteration of MIMO training."""
import os, sys, json, time
os.chdir("/workspace/Megatron-LM/.worktrees/nmfw-23-benchmarking")

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function

# Init distributed
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
rank = dist.get_rank()

# Load config
from benchmarks.mimo_throughput.config_loader import load_config
from benchmarks.mimo_throughput.training import create_mimo_model, forward_step, loss_func, ENCODER_NAME
from benchmarks.mimo_throughput.data import SyntheticVLMIterator
from benchmarks.mimo_throughput.process_groups import ProcessGroupManager
from functools import partial

config = load_config("benchmarks/mimo_throughput/configs/1node_1b_7b/hetero_enc_tp2dp4_llm_tp4dp2.yaml")
config.validate()

# Create model
pg_manager = ProcessGroupManager()
mimo_model, ctx = create_mimo_model(config, pg_manager)
encoder_grid = ctx['encoder_grid']
llm_grid = ctx['llm_grid']
llm_pg = ctx['llm_pg']

data_iter = SyntheticVLMIterator(
    encoder_hidden_size=config.encoder_arch.hidden_size,
    image_seq_length=config.encoder_arch.seq_length,
    total_seq_length=config.llm_arch.seq_length,
    micro_batch_size=config.data.micro_batch_size,
    vocab_size=config.llm_arch.vocab_size,
    image_token_id=config.data.image_token_id,
    encoder_name=ENCODER_NAME,
)

from megatron.core.pipeline_parallel import schedules

# Warmup 2 iterations
for _ in range(2):
    schedules.forward_backward_no_pipelining(
        forward_step_func=partial(forward_step, encoder_grid=encoder_grid, llm_grid=llm_grid, encoder_name=ENCODER_NAME),
        data_iterator=data_iter, model=[mimo_model],
        num_microbatches=config.data.num_microbatches,
        seq_length=config.llm_arch.seq_length, micro_batch_size=config.data.micro_batch_size,
        forward_only=False, pg_collection=llm_pg,
    )
torch.cuda.synchronize()

# Profile 1 iteration with RICH settings
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    schedules.forward_backward_no_pipelining(
        forward_step_func=partial(forward_step, encoder_grid=encoder_grid, llm_grid=llm_grid, encoder_name=ENCODER_NAME),
        data_iterator=data_iter, model=[mimo_model],
        num_microbatches=config.data.num_microbatches,
        seq_length=config.llm_arch.seq_length, micro_batch_size=config.data.micro_batch_size,
        forward_only=False, pg_collection=llm_pg,
    )

if rank == 0:
    # 1. Standard table
    print("=" * 80)
    print("TABLE (cuda_time_total, top 30):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # 2. Memory table
    print("=" * 80)
    print("TABLE (self_device_memory_usage, top 20):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_device_memory_usage", row_limit=20))

    # 3. Extract structured data
    events = prof.key_averages()
    
    # Categorize ops
    compute_time = 0
    comm_time = 0
    memory_ops_time = 0
    other_time = 0
    total_cuda = 0
    
    nccl_breakdown = {}
    top_memory_allocs = []
    
    for e in events:
        cuda_us = e.cuda_time_total
        total_cuda += cuda_us
        name = e.key
        
        # Categorize
        if any(k in name.lower() for k in ['nccl', 'allreduce', 'allgather', 'reduce_scatter', 'broadcast']):
            comm_time += cuda_us
            # Further breakdown by type
            for ctype in ['AllReduce', 'AllGather', 'ReduceScatter', 'Broadcast']:
                if ctype.lower() in name.lower():
                    nccl_breakdown[ctype] = nccl_breakdown.get(ctype, 0) + cuda_us
        elif any(k in name.lower() for k in ['gemm', 'nvjet', 'cutlass', 'cublas', 'linear', 'attn', 'fused']):
            compute_time += cuda_us
        elif any(k in name.lower() for k in ['memcpy', 'memset']):
            memory_ops_time += cuda_us
        else:
            other_time += cuda_us
        
        # Track memory
        mem = e.self_device_memory_usage
        if mem > 0:
            top_memory_allocs.append((name, mem, e.count))
    
    top_memory_allocs.sort(key=lambda x: -x[1])
    
    print("=" * 80)
    print("STRUCTURED ANALYSIS:")
    print("=" * 80)
    print(f"\nTotal CUDA time: {total_cuda/1e6:.1f}s")
    print(f"  Compute (GEMM/Attn/Fused): {compute_time/1e6:.3f}s ({100*compute_time/max(total_cuda,1):.1f}%)")
    print(f"  Communication (NCCL):       {comm_time/1e6:.3f}s ({100*comm_time/max(total_cuda,1):.1f}%)")
    print(f"  Memory ops:                 {memory_ops_time/1e6:.3f}s ({100*memory_ops_time/max(total_cuda,1):.1f}%)")
    print(f"  Other:                      {other_time/1e6:.3f}s ({100*other_time/max(total_cuda,1):.1f}%)")
    
    print(f"\nNCCL breakdown:")
    for ctype, us in sorted(nccl_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {ctype}: {us/1e3:.1f}ms ({100*us/max(total_cuda,1):.1f}%)")
    
    print(f"\nTop memory allocations:")
    for name, mem_bytes, count in top_memory_allocs[:15]:
        print(f"  {name}: {mem_bytes/1e6:.1f}MB ({count} calls)")
    
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"Current GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved GPU memory: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Cleanup
del mimo_model
torch.cuda.empty_cache()
pg_manager.destroy_all()
dist.destroy_process_group()
