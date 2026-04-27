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

from contextlib import contextmanager
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel import schedules


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

    # ── Phase 1: Encoder forward on full batch (one pass) ────────────────
    # All ranks participate (encoder is PP=1, communicate is collective).
    all_batches = [next(data_iterator) for _ in range(num_microbatches)]
    full_encoder_input = _concat_encoder_inputs(all_batches, encoder_name)
    _slice_for_encoder_dp(full_encoder_input, encoder_grid, llm_grid)

    enc_out = mimo_model.encode_and_communicate({encoder_name: full_encoder_input})

    # Detach so Phase 2 runs no encoder collectives; microbatch views accumulate
    # .grad into detached_full.grad automatically.
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

    # Swap in a capturing finalize so the inner PP schedule does not run DDP
    # grad sync before Phase 3 has produced encoder grads. The capture also
    # records ``num_tokens`` that the inner schedule would have passed — we
    # forward it to the original finalize after Phase 3 so per-token-loss
    # configs see the correct global divisor.
    with _deferred_finalize(mimo_model.config) as (original_finalize, capture):
        losses = schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=_lm_forward_step,
            data_iterator=cache_iter,
            model=[mimo_model],
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            **schedule_kwargs,
        )

    # ── Phase 3: Encoder backward (one pass, all ranks sync) ────────────
    # detached_full.grad was populated by Phase 2's per-microbatch LLM backward
    # (accumulated across microbatch view slices on PP stage 0).
    # Broadcast to PP stage 1+ then run one encoder backward for the full batch.
    if not forward_only and enc_out:
        _broadcast_encoder_grad(detached_full, enc_out, pp_group, is_pp_first)
        for key in enc_out:
            grad = detached_full[key].grad
            if grad is not None:
                torch.autograd.backward(enc_out[key], grad_tensors=grad)

    # Single post-Phase-3 finalize: reduces LLM grads (from Phase 2) and
    # encoder grads (from Phase 3) together. Without this call, encoder
    # grads remain local to each rank and Adam steps on un-reduced grads,
    # causing silent divergence from the equal-DP reference.
    if not forward_only and original_finalize is not None:
        original_finalize(
            [mimo_model],
            capture.num_tokens,
            pg_collection=schedule_kwargs.get('pg_collection'),
            force_all_reduce=False,
        )

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
    """Default loss function for the LLM pipeline.

    Returns the 3-tuple ``(local_sum, local_num_tokens, log_dict)`` contract
    expected when ``calculate_per_token_loss=True`` is set on the
    TransformerConfig. When it is not set, the schedule divides
    ``local_sum`` by ``local_num_tokens`` (clamped to 1), so the 3-tuple
    form is also safe for standard per-microbatch-mean configs.
    """
    if output_tensor is None:
        zero_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        zero_count = torch.tensor(0, device='cuda', dtype=torch.int)
        return zero_loss, zero_count, {'loss_reduced': 0.0}
    masked = output_tensor.float() * loss_mask.float()
    local_sum = masked.sum()
    local_num_tokens = loss_mask.float().sum().to(torch.int)
    return local_sum, local_num_tokens, {'loss_reduced': local_sum.detach().item()}


class _CapturingFinalize:
    """Capture the ``num_tokens`` the inner PP schedule would have passed.

    The three-phase schedule defers grad finalization until after Phase 3
    runs encoder backward. Replacing the config's ``finalize_model_grads_func``
    with this object absorbs the inner schedule's invocation and stores
    ``num_tokens`` so the post-Phase-3 call to the original finalize can
    forward it — required for ``calculate_per_token_loss=True`` configs
    whose finalize hook divides by the global valid-token count.
    """

    def __init__(self):
        self.num_tokens = None

    def __call__(self, model_list, num_tokens, *args, **kwargs):
        self.num_tokens = num_tokens
        return None


@contextmanager
def _deferred_finalize(config):
    """Suppress the PP schedule's end-of-run DDP grad sync; yield the
    original finalize and a capture object so callers can invoke the
    original (with the captured ``num_tokens``) once after Phase 3.
    """
    original = config.finalize_model_grads_func
    capture = _CapturingFinalize()
    config.finalize_model_grads_func = capture
    try:
        yield original, capture
    finally:
        config.finalize_model_grads_func = original
