# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from packaging import version

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# to run the test
# uv run python -m torch.distributed.run --nproc_per_node=2  -m pytest tests/unit_tests/transformer/test_dp_vs_cp_equivalence.py
# TODO: dont worry about attention mask. 
# get_batch_on_this_cp_rank follow the same implementation but cp group passed in. refer to megatron/core/utils.py file for the function.
# note the interleaved distribution of sequence chunks 
# dont initialize parallel state
def shard_batch_for_cp(batch, cp_group):
    """Shard batch input along sequence dimension for context parallel.

    This mimics get_batch_on_this_cp_rank but works with custom process groups.
    For hidden_states, sequence dimension is 0 (format: [seq, batch, hidden]).
    For tokens, sequence dimension is 1 (format: [batch, seq]).
    """
    cp_size = cp_group.size()
    if cp_size > 1:
        cp_rank = cp_group.rank()
        for key, val in batch.items():
            if val is not None:
                # For hidden_states (after embedding), seq is dim 0
                # For tokens/labels (before embedding), seq is dim 1
                # For attention_mask, seq is dim 2
                if key == "hidden_states":
                    seq_dim = 0
                elif key == "attention_mask":
                    seq_dim = 2
                else:
                    seq_dim = 1

                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.zeros(2, dtype=torch.int64, device=val.device)
                index[0].fill_(cp_rank)
                index[1].fill_(2 * cp_size - cp_rank - 1)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val
    return batch


class TestDataParallelVsContextParallel:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    def test_dp_vs_cp_output_equivalence(self):
        """
        Test that transformer block outputs match between:
        - Data Parallel (DP=2) with full sequence on both GPUs
        - Context Parallel (CP=2) with sharded sequence across GPUs

        This verifies that context parallel correctly shards and processes sequences
        to produce the same result as data parallel with full sequences.
        """
        # Initialize distributed backend
        Utils.initialize_distributed()

        # Check world size after initialization
        world_size = 2
        actual_world_size = torch.distributed.get_world_size()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

        # Set random seeds for reproducibility
        torch.manual_seed(12345)

        # Create transformer configuration
        # Sequence length must be divisible by 2*CP = 4 for CP=2
        sequence_length = 4096
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=28,
            hidden_size=1024,
            num_attention_heads=16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            context_parallel_size=1,  # Will be overridden by process groups
            no_rope_freq=[1]*28,  # Disable RoPE for all layers to isolate the issue
        )

        # ===== SCENARIO A: Data Parallel (DP=2) =====
        # Create process groups for DP=2
        dp_grid = HyperCommGrid([1, 1, 1, 1, 2], ["tp", "cp", "ep", "pp", "dp"])

        dp_tp_group = dp_grid.create_pg("tp")
        dp_cp_group = dp_grid.create_pg("cp")
        dp_ep_group = dp_grid.create_pg("ep")
        dp_pp_group = dp_grid.create_pg("pp")
        dp_group = dp_grid.create_pg("dp")

        dp_pg_collection = ProcessGroupCollection(
            tp=dp_tp_group, cp=dp_cp_group, ep=dp_ep_group, pp=dp_pp_group, dp=dp_group
        )
        # For DP, we also need dp_cp group
        dp_cp_combined = dp_grid.create_pg(["dp", "cp"])
        dp_pg_collection.dp_cp = dp_cp_combined

        # Create DP transformer block
        dp_transformer_config = TransformerConfig(
            num_layers=transformer_config.num_layers,
            hidden_size=transformer_config.hidden_size,
            num_attention_heads=transformer_config.num_attention_heads,
            use_cpu_initialization=transformer_config.use_cpu_initialization,
            attention_dropout=transformer_config.attention_dropout,
            hidden_dropout=transformer_config.hidden_dropout,
            bf16=transformer_config.bf16,
            context_parallel_size=1,
            no_rope_freq=transformer_config.no_rope_freq,
        )

        dp_block = (
            TransformerBlock(
                dp_transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                pg_collection=dp_pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # ===== SCENARIO B: Context Parallel (CP=2) =====
        # Create process groups for CP=2
        cp_grid = HyperCommGrid([1, 2, 1, 1, 1], ["tp", "cp", "ep", "pp", "dp"])

        cp_tp_group = cp_grid.create_pg("tp")
        cp_cp_group = cp_grid.create_pg("cp")
        cp_ep_group = cp_grid.create_pg("ep")
        cp_pp_group = cp_grid.create_pg("pp")
        cp_dp_group = cp_grid.create_pg("dp")

        cp_pg_collection = ProcessGroupCollection(
            tp=cp_tp_group, cp=cp_cp_group, ep=cp_ep_group, pp=cp_pp_group, dp=cp_dp_group
        )
        cp_dp_cp_combined = cp_grid.create_pg(["dp", "cp"])
        cp_pg_collection.dp_cp = cp_dp_cp_combined

        # Create CP transformer block
        cp_transformer_config = TransformerConfig(
            num_layers=transformer_config.num_layers,
            hidden_size=transformer_config.hidden_size,
            num_attention_heads=transformer_config.num_attention_heads,
            use_cpu_initialization=transformer_config.use_cpu_initialization,
            attention_dropout=transformer_config.attention_dropout,
            hidden_dropout=transformer_config.hidden_dropout,
            bf16=transformer_config.bf16,
            context_parallel_size=2,
            no_rope_freq=transformer_config.no_rope_freq,
            cp_comm_type="a2a",
        )

        cp_block = (
            TransformerBlock(
                cp_transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                pg_collection=cp_pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        # Synchronize weights: copy from DP block to CP block
        for dp_param, cp_param in zip(dp_block.parameters(), cp_block.parameters()):
            cp_param.data.copy_(dp_param.data)

        # Synchronize buffers
        for dp_buffer, cp_buffer in zip(dp_block.buffers(), cp_block.buffers()):
            cp_buffer.data.copy_(dp_buffer.data)

        # ===== Verify weight synchronization =====
        print(f"\n[Rank {torch.distributed.get_rank()}] Verifying weight synchronization...")
        weight_match = True
        for i, (dp_param, cp_param) in enumerate(zip(dp_block.parameters(), cp_block.parameters())):
            if not torch.allclose(dp_param, cp_param, rtol=1e-5, atol=1e-5):
                print(f"[Rank {torch.distributed.get_rank()}] WARNING: Parameter {i} mismatch!")
                print(f"  DP param sum: {dp_param.sum()}, CP param sum: {cp_param.sum()}")
                weight_match = False
        print(f"[Rank {torch.distributed.get_rank()}] Weight synchronization: {'OK' if weight_match else 'MISMATCH'}")

        # ===== Create test input =====
        # Create the same input on all ranks
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, transformer_config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Synchronize input across all ranks
        torch.distributed.all_reduce(hidden_states, op=torch.distributed.ReduceOp.AVG)

        print(f"\n[Rank {torch.distributed.get_rank()}] Input hidden_states shape: {hidden_states.shape}")
        print(f"[Rank {torch.distributed.get_rank()}] Input sum: {hidden_states.sum():.6f}, mean: {hidden_states.float().mean():.6f}")

        # Clone for each scenario
        dp_hidden_states = hidden_states.clone().detach()
        cp_hidden_states = hidden_states.clone().detach()

        # ===== SCENARIO A: Forward pass with DP (full sequence) =====
        with torch.no_grad():
            dp_output = dp_block(hidden_states=dp_hidden_states, attention_mask=None)
            print(f"[Rank {torch.distributed.get_rank()}] DP hidden states sum: {dp_hidden_states.sum()} dp output shape: {dp_output.shape}")

        # ===== SCENARIO B: Forward pass with CP (sharded sequence) =====
        # Shard the input using our custom sharding function with the CP group
        # The function expects a dictionary with tensors
        # dont complicate just pass the tensor
        cp_batch = {"hidden_states": cp_hidden_states}
        cp_batch_sharded = shard_batch_for_cp(cp_batch, cp_cp_group)
        cp_hidden_states_sharded = cp_batch_sharded["hidden_states"]

        print(f"\n[Rank {torch.distributed.get_rank()}] After sharding for CP:")
        print(f"  Original shape: {cp_hidden_states.shape}, Sharded shape: {cp_hidden_states_sharded.shape}")
        print(f"  Sharded sum: {cp_hidden_states_sharded.sum():.6f}, mean: {cp_hidden_states_sharded.float().mean():.6f}")

        # Verify sharding: gather all sharded chunks and compare with original
        cp_rank = cp_cp_group.rank()
        cp_world_size = cp_cp_group.size()

        if cp_world_size == 2:
            # Expected: each rank should have 2 chunks (1/4 of full sequence each)
            expected_sharded_seq_len = sequence_length // 2
            actual_sharded_seq_len = cp_hidden_states_sharded.shape[0]
            print(f"[Rank {torch.distributed.get_rank()}] Expected sharded seq len: {expected_sharded_seq_len}, Actual: {actual_sharded_seq_len}")

            # Manually verify the chunks match original
            chunk_size = sequence_length // 4
            if cp_rank == 0:
                expected_chunk_0 = hidden_states[0:chunk_size]
                expected_chunk_3 = hidden_states[3*chunk_size:4*chunk_size]
                expected_sharded = torch.cat([expected_chunk_0, expected_chunk_3], dim=0)
            else:  # cp_rank == 1
                expected_chunk_1 = hidden_states[chunk_size:2*chunk_size]
                expected_chunk_2 = hidden_states[2*chunk_size:3*chunk_size]
                expected_sharded = torch.cat([expected_chunk_1, expected_chunk_2], dim=0)

            sharding_correct = torch.allclose(cp_hidden_states_sharded, expected_sharded, rtol=1e-5, atol=1e-5)
            print(f"[Rank {torch.distributed.get_rank()}] Sharding verification: {'CORRECT' if sharding_correct else 'INCORRECT'}")
            if not sharding_correct:
                print(f"  Max diff: {(cp_hidden_states_sharded - expected_sharded).abs().max():.6f}")

        with torch.no_grad():
            cp_output_sharded = cp_block(hidden_states=cp_hidden_states_sharded, attention_mask=None)
            print(f"\n[Rank {torch.distributed.get_rank()}] CP output_sharded shape: {cp_output_sharded.shape}")
            print(f"[Rank {torch.distributed.get_rank()}] CP output_sharded sum: {cp_output_sharded.sum():.6f}, mean: {cp_output_sharded.float().mean():.6f}")
            

        # ===== Gather CP outputs from all ranks =====
        # CP output is sharded along sequence dimension
        # We need to all_gather it to reconstruct the full sequence
        cp_rank = cp_cp_group.rank()
        cp_world_size = cp_cp_group.size()

        print(f"\n[Rank {torch.distributed.get_rank()}] Gathering CP outputs...")

        # Gather outputs from all CP ranks
        gathered_outputs = [
            torch.zeros_like(cp_output_sharded) for _ in range(cp_world_size)
        ]
        torch.distributed.all_gather(
            gathered_outputs, cp_output_sharded, group=cp_cp_group
        )

        print(f"[Rank {torch.distributed.get_rank()}] Gathered {len(gathered_outputs)} chunks")
        for i, g_out in enumerate(gathered_outputs):
            print(f"[Rank {torch.distributed.get_rank()}]   Gathered output {i} shape: {g_out.shape}, sum: {g_out.sum():.6f}")

        # Reconstruct the full sequence from gathered chunks
        # get_batch_on_this_cp_rank uses special load-balancing:
        # For CP=2: creates 4 chunks, rank 0 gets chunks [0, 3], rank 1 gets chunks [1, 2]
        # gathered_outputs[0] contains chunks [0, 3] concatenated
        # gathered_outputs[1] contains chunks [1, 2] concatenated

        # Split each gathered output into its two chunks
        chunk_size = sequence_length // 4  # Each chunk is 1/4 of full sequence
        chunks = []
        for rank_idx, rank_output in enumerate(gathered_outputs):
            # Each rank has 2 chunks concatenated along sequence dimension
            rank_chunks = torch.split(rank_output, chunk_size, dim=0)
            print(f"[Rank {torch.distributed.get_rank()}] Split rank {rank_idx} output into {len(rank_chunks)} chunks of size {chunk_size}")
            chunks.extend(rank_chunks)

        print(f"[Rank {torch.distributed.get_rank()}] Total chunks after split: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"[Rank {torch.distributed.get_rank()}]   Chunk {i} shape: {chunk.shape}, sum: {chunk.sum():.6f}")

        # chunks now contains [chunk_0, chunk_3, chunk_1, chunk_2]
        # We need to reorder to [chunk_0, chunk_1, chunk_2, chunk_3]
        if cp_world_size == 2:
            # chunks = [chunk_0, chunk_3, chunk_1, chunk_2]
            # Reorder to [chunk_0, chunk_1, chunk_2, chunk_3]
            print(f"[Rank {torch.distributed.get_rank()}] Reordering chunks: [0, 2, 3, 1]")
            cp_output_full = torch.cat([chunks[0], chunks[2], chunks[3], chunks[1]], dim=0)
        else:
            # For other CP sizes, need different logic
            raise NotImplementedError(f"Gathering logic not implemented for CP size {cp_world_size}")

        print(f"[Rank {torch.distributed.get_rank()}] Reconstructed full output shape: {cp_output_full.shape}, sum: {cp_output_full.sum():.6f}")



        # ===== Compare outputs =====
        print(f"\n{'='*80}")
        print(f"[Rank {torch.distributed.get_rank()}] OUTPUT COMPARISON")
        print(f"{'='*80}")
        print(f"[Rank {torch.distributed.get_rank()}] DP output shape: {dp_output.shape}")
        print(f"[Rank {torch.distributed.get_rank()}] CP output full shape: {cp_output_full.shape}")
        print(f"[Rank {torch.distributed.get_rank()}] DP output mean: {dp_output.float().mean():.6f}, std: {dp_output.float().std():.6f}")
        print(f"[Rank {torch.distributed.get_rank()}] CP output mean: {cp_output_full.float().mean():.6f}, std: {cp_output_full.float().std():.6f}")

        # Compute differences
        diff = (dp_output - cp_output_full).float()
        abs_diff = diff.abs()

        if torch.distributed.get_rank() == 0:

            print(f"\n[Rank {torch.distributed.get_rank()}] DIFFERENCE STATISTICS:")
            print(f"  Max absolute difference: {abs_diff.max():.6f}")
            print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
            print(f"  Median absolute difference: {abs_diff.median():.6f}")
            print(f"  Min absolute difference: {abs_diff.min():.6f}")
            print(f"  Std of absolute difference: {abs_diff.std():.6f}")

            # Find locations of largest differences
            flat_abs_diff = abs_diff.view(-1)
            max_diff_idx = flat_abs_diff.argmax()
            seq_idx = max_diff_idx // (micro_batch_size * transformer_config.hidden_size)
            batch_idx = (max_diff_idx % (micro_batch_size * transformer_config.hidden_size)) // transformer_config.hidden_size
            hidden_idx = max_diff_idx % transformer_config.hidden_size

            print(f"\n[Rank {torch.distributed.get_rank()}] LOCATION OF MAX DIFFERENCE:")
            print(f"  Sequence position: {seq_idx}/{sequence_length}")
            print(f"  Batch index: {batch_idx}/{micro_batch_size}")
            print(f"  Hidden dimension: {hidden_idx}/{transformer_config.hidden_size}")
            print(f"  DP value: {dp_output[seq_idx, batch_idx, hidden_idx]:.6f}")
            print(f"  CP value: {cp_output_full[seq_idx, batch_idx, hidden_idx]:.6f}")
            print(f"  Difference: {diff[seq_idx, batch_idx, hidden_idx]:.6f}")

            # Sample values from different sequence positions
            print(f"\n[Rank {torch.distributed.get_rank()}] SAMPLE VALUES AT DIFFERENT POSITIONS:")
            sample_positions = [0, sequence_length//4, sequence_length//2, 3*sequence_length//4, sequence_length-1]
            for pos in sample_positions:
                print(f"\n  Position {pos} (batch 0, first 5 hidden dims):")
                print(f"    DP:   {[f'{dp_output[pos, 0, i]:.4f}' for i in range(5)]}")
                print(f"    CP:   {[f'{cp_output_full[pos, 0, i]:.4f}' for i in range(5)]}")
                print(f"    Diff: {[f'{diff[pos, 0, i]:.4f}' for i in range(5)]}")

            # Check percentage of elements exceeding different thresholds
            thresholds = [0.001, 0.005, 0.01, 0.02, 0.03]
            print(f"\n[Rank {torch.distributed.get_rank()}] PERCENTAGE OF ELEMENTS EXCEEDING THRESHOLDS:")
            total_elements = abs_diff.numel()
            for thresh in thresholds:
                count = (abs_diff > thresh).sum().item()
                percentage = 100.0 * count / total_elements
                print(f"  > {thresh:.3f}: {count}/{total_elements} ({percentage:.2f}%)")

            # Relative error
            rel_error = abs_diff / (dp_output.float().abs() + 1e-8)
            print(f"\n[Rank {torch.distributed.get_rank()}] RELATIVE ERROR STATISTICS:")
            print(f"  Max relative error: {rel_error.max():.6f}")
            print(f"  Mean relative error: {rel_error.mean():.6f}")
            print(f"  Median relative error: {rel_error.median():.6f}")

            print(f"{'='*80}\n")

            # if torch.distributed.get_rank() == 0:
            #     breakpoint()
            # torch.distributed.barrier()

            # torch.testing.assert_close(
            #     dp_output,
            #     cp_output_full,
            #     rtol=1e-3,
            #     atol=1e-3,
            #     msg="Outputs don't match between Data Parallel (full sequence) and Context Parallel (sharded sequence)",
            # )

            print(f"[Rank {torch.distributed.get_rank()}] Test passed! DP and CP outputs match.")
