# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Three-phase schedule for colocated MIMO training with LLM PP>1.

Phase 1: Encoder forward + communicate for the full batch (all ranks synchronized).
Phase 2: LLM 1F1B pipeline with detached encoder embeddings sliced per microbatch.
Phase 3: Encoder backward for the full batch (all ranks synchronized).

Encoder runs on all ranks (PP=1) and its TP/DP collectives require all ranks
to participate simultaneously. The 1F1B pipeline staggers ranks across PP stages,
so encoder collectives cannot run inside the pipeline. The three-phase design
separates encoder (synchronized) from LLM (pipelined) by detaching the autograd
graph at the encoder-LLM boundary.
"""

from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)


def colocated_forward_backward_with_pp(
    mimo_model,
    data_iterator,
    num_microbatches: int,
    encoder_grid: Optional[HyperCommGrid] = None,
    llm_grid: Optional[HyperCommGrid] = None,
    encoder_name: str = "images",
    forward_only: bool = False,
    **schedule_kwargs,
):
    """Three-phase colocated training: encoder batch -> LLM pipeline -> encoder backward.

    Args:
        mimo_model: MimoModel with colocated communicators and lm_has_pp=True.
        data_iterator: Yields dicts with input_ids, labels, etc.
        num_microbatches: Number of microbatches for the LLM pipeline.
        encoder_grid: Encoder HyperCommGrid (for DP fan-in slicing).
        llm_grid: LLM HyperCommGrid (for PP group).
        encoder_name: Modality name for the encoder (e.g., "images").
        forward_only: Skip backward passes if True.
        **schedule_kwargs: Passed to forward_backward_pipelining_without_interleaving.
            Must include p2p_communicator, pg_collection, seq_length, micro_batch_size.
    """
    pp_group = llm_grid.get_pg("pp") if llm_grid and 'pp' in llm_grid.dim_names else None
    is_pp_first = pp_group is None or pp_group.rank() == 0

    with record_function("mimo::forward_backward"):
        # ── Phase 1: Encoder forward on full batch (one pass) ────────────────
        # All ranks participate (encoder is PP=1, communicate is collective).
        all_batches = [next(data_iterator) for _ in range(num_microbatches)]
        full_encoder_input = _concat_encoder_inputs(all_batches, encoder_name)
        _slice_for_encoder_dp(full_encoder_input, encoder_grid, llm_grid)

        # Initialize offload manager for MIMO-level offloading in Phase 1.
        # encode_and_communicate doesn't go through _forward_all_modules,
        # so we must init the chunk handler here.
        has_mimo_offloading = getattr(mimo_model, '_has_mimo_offloading', False)
        if has_mimo_offloading and mimo_model.training:
            off_interface.init_chunk_handler(
                vp_size=1, vp_stage=None, min_offloaded_tensor_size=1024
            )

        with record_function("mimo::encoder_forward"):
            enc_out = mimo_model.encode_and_communicate({encoder_name: full_encoder_input})

        # Detach: sever autograd link to encoder so Phase 2 has no encoder collectives.
        # Microbatch slices are views into detached_full — their .grad accumulates
        # into detached_full.grad automatically via PyTorch's view gradient semantics.
        detached_full = {k: v.detach().requires_grad_(True) for k, v in enc_out.items()}
        lm_data = _build_lm_microbatches(detached_full, all_batches, num_microbatches)

        # ── Phase 2: LLM 1F1B pipeline ──────────────────────────────────────
        # Only LLM P2P communication (within PP group). No encoder collectives.
        cache_iter = iter(lm_data)

        def _lm_forward_step(data_iterator_unused, model, *args):
            cached = next(cache_iter)
            output_tensor, loss_mask = model(
                input_ids=cached['input_ids'],
                labels=cached['labels'],
                loss_mask=cached['loss_mask'],
                position_ids=cached['position_ids'],
                encoder_embeddings=cached['encoder_embeddings'],
            )
            return output_tensor, partial(_loss_func, cached['loss_mask'])

        # Suppress the schedule's end-of-iteration off_interface.reset() — Phase 1
        # may have offloaded tensors that Phase 3's encoder backward still needs.
        # We use a _suppress_offload_reset flag checked nowhere else; the schedule
        # checks config.fine_grained_activation_offloading which we must NOT mutate
        # here because GPTModel.forward() also reads it for init_chunk_handler.
        config = mimo_model.config
        _saved_offload_flag = getattr(config, 'fine_grained_activation_offloading', False)
        if _saved_offload_flag:
            config.fine_grained_activation_offloading = False

        try:
            with record_function("mimo::llm_forward"):
                losses = schedules.forward_backward_pipelining_without_interleaving(
                    forward_step_func=_lm_forward_step,
                    data_iterator=cache_iter,
                    model=[mimo_model],
                    num_microbatches=num_microbatches,
                    forward_only=forward_only,
                    **schedule_kwargs,
                )

            # ── Phase 3: Encoder backward (one pass, all ranks sync) ────────
            if not forward_only and enc_out:
                _broadcast_encoder_grad(detached_full, enc_out, pp_group, is_pp_first)
                for key in enc_out:
                    grad = detached_full[key].grad
                    if grad is not None:
                        torch.autograd.backward(enc_out[key], grad_tensors=grad)
        finally:
            # Restore flag and reset offload manager after Phase 3 completes
            # (or on exception, to avoid leaking CPU pinned memory).
            config.fine_grained_activation_offloading = _saved_offload_flag
            if _saved_offload_flag:
                off_interface.reset()

    return losses


# ── Helpers ──────────────────────────────────────────────────────────────


def _concat_encoder_inputs(all_batches, encoder_name):
    """Concatenate encoder inputs from all microbatches along batch dim (dim 1)."""
    first = all_batches[0]
    result = {}
    if not (first.get('modality_inputs') and encoder_name in first['modality_inputs']):
        return result
    for enc_name in first['modality_inputs'][encoder_name]:
        result[enc_name] = {}
        for key in first['modality_inputs'][encoder_name][enc_name]:
            vals = [
                b['modality_inputs'][encoder_name][enc_name][key]
                for b in all_batches
                if b.get('modality_inputs') and encoder_name in b['modality_inputs']
            ]
            tensors = [v for v in vals if isinstance(v, torch.Tensor)]
            result[enc_name][key] = torch.cat(tensors, dim=1) if tensors else vals[0]
    return result


def _slice_for_encoder_dp(full_encoder_input, encoder_grid, llm_grid):
    """Slice concatenated encoder input for fan-in (enc_dp > llm_dp)."""
    if encoder_grid is None or llm_grid is None:
        return
    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()
    if enc_dp <= llm_dp:
        return
    scale = enc_dp // llm_dp
    slot = encoder_grid.get_pg("dp").rank() % scale
    for enc_name in full_encoder_input:
        for key, tensor in full_encoder_input[enc_name].items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                bs = tensor.shape[1]
                ss = bs // scale
                if ss == 0:
                    raise ValueError(
                        f"Encoder fan-in produces zero-sized batch: "
                        f"total_batch={bs}, scale={scale}. Increase micro_batch_size."
                    )
                full_encoder_input[enc_name][key] = tensor[
                    :, slot * ss : (slot + 1) * ss, :
                ].contiguous()


def _build_lm_microbatches(detached_full, all_batches, num_microbatches):
    """Slice detached encoder output into per-microbatch views for the LLM pipeline."""
    if not detached_full:
        # Text-only batch: no encoder embeddings to slice
        return [
            {
                'encoder_embeddings': {},
                'input_ids': all_batches[mb].get('input_ids'),
                'labels': all_batches[mb].get('labels'),
                'loss_mask': all_batches[mb].get('loss_mask'),
                'position_ids': all_batches[mb].get('position_ids'),
            }
            for mb in range(num_microbatches)
        ]

    sample = next(iter(detached_full.values()))
    batch_dim = 1 if sample.ndim == 3 else 0
    total_batch = sample.shape[batch_dim]
    assert total_batch % num_microbatches == 0, (
        f"Encoder output batch ({total_batch}) must be divisible "
        f"by num_microbatches ({num_microbatches})"
    )
    mb_size = total_batch // num_microbatches

    lm_data = []
    for mb in range(num_microbatches):
        s, e = mb * mb_size, (mb + 1) * mb_size
        mb_enc = {}
        for k, v in detached_full.items():
            mb_enc[k] = v[:, s:e, :] if v.ndim == 3 else v[s:e, :]
        lm_data.append(
            {
                'encoder_embeddings': mb_enc,
                'input_ids': all_batches[mb].get('input_ids'),
                'labels': all_batches[mb].get('labels'),
                'loss_mask': all_batches[mb].get('loss_mask'),
                'position_ids': all_batches[mb].get('position_ids'),
            }
        )
    return lm_data


def _broadcast_encoder_grad(detached_full, enc_out, pp_group, is_pp_first):
    """Broadcast encoder gradient from PP stage 0 to stage 1+ ranks."""
    if pp_group is None or pp_group.size() <= 1:
        return
    src = dist.get_global_rank(pp_group, 0)
    for key in enc_out:
        if is_pp_first:
            assert (
                detached_full[key].grad is not None
            ), f"No encoder gradient on PP stage 0 for '{key}'"
            dist.broadcast(detached_full[key].grad, src=src, group=pp_group)
        else:
            grad = torch.zeros_like(detached_full[key])
            dist.broadcast(grad, src=src, group=pp_group)
            detached_full[key].grad = grad


def _loss_func(loss_mask, output_tensor):
    """Default loss function for the LLM pipeline."""
    if output_tensor is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}
    loss = output_tensor.float().sum()
    return loss, {'loss_reduced': loss.detach().item()}
