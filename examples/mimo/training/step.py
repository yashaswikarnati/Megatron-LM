# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Forward step and per-token loss for MIMO training on the stock schedule.

This module provides the ``forward_step_func`` that the stock Megatron
schedule (``forward_backward_pipelining_without_interleaving``) calls. It is
role-agnostic: ``MimoModel.forward`` routes the call to the encoder or the
language module depending on the rank's role, and this module only needs to
pull a batch, move it to CUDA, run the model, and return the per-token loss
closure.

The contract with the stock schedule (``calculate_per_token_loss=True``) is:

  ``forward_step_func(data_iterator, model) -> (output_tensor, loss_func)``

and ``loss_func(output_tensor)`` returns the 3-tuple
``(loss_sum, num_tokens, loss_dict)`` where ``num_tokens`` is an *integer*
tensor (the schedule accumulates it into an int ``total_num_tokens``; a float
triggers ``RuntimeError: result type Float can't be cast to the desired
output type Int`` on last-PP-stage ranks, which deadlocks the collective).

Scope note: this is the NON-COLOCATED path. The encoder-prefetch fast-path
from the prototype (``forward_step_encode_only`` + prefetcher) is intentionally
omitted here.
"""

from __future__ import annotations

from functools import partial

import torch


def loss_func(output_tensor: torch.Tensor, *, loss_mask: torch.Tensor):
    """Return terminal language-model loss sum, local token count, and logging tensors.

    Returns the stock per-token-loss 3-tuple ``(loss_sum, num_tokens,
    loss_dict)``:

    * ``loss_sum`` -- summed (NOT averaged) per-token loss over the masked
      tokens. The schedule / grad-finalization owns the division by the global
      token count.
    * ``num_tokens`` -- an *integer* tensor. The schedule accumulates this into
      an int ``total_num_tokens``; a float dtype here casts-faults on the
      last-PP-stage ranks and deadlocks.
    * ``loss_dict`` -- ``{"lm loss": stack(loss_sum, num_tokens)}`` for the
      logging all-reduce, matching the prototype.
    """
    if output_tensor is None:
        raise RuntimeError("terminal language stage returned no loss tensor")
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(
            "loss_func expects the terminal language stage to return a tensor, "
            f"got {type(output_tensor).__name__}"
        )

    output = output_tensor.float()
    if loss_mask is None:
        raise RuntimeError("MIMO training requires a loss_mask for per-token loss")
    if output.shape != loss_mask.shape:
        raise RuntimeError(
            f"loss output shape {tuple(output.shape)} does not match loss_mask shape "
            f"{tuple(loss_mask.shape)}; per-token loss cannot be scaled correctly"
        )

    masked = output * loss_mask.float()
    # num_tokens MUST be an integer tensor for the stock schedule's int
    # total_num_tokens accumulation.
    num_tokens = loss_mask.float().sum().to(torch.int)
    loss_sum = masked.sum()
    return (
        loss_sum,
        num_tokens,
        {"lm loss": torch.stack((loss_sum.detach(), num_tokens.detach().float()))},
    )


def mimo_forward_step(data_iterator, model):
    """Forward step consumed by the stock MCore pipeline schedule.

    Role-agnostic: pulls a batch, moves it to CUDA, and calls ``model(**batch)``.
    The role routing (encoder vs language module vs colocated) lives inside
    ``MimoModel.forward``, which returns ``(output, loss_mask)`` for the
    non-colocated path.

    Returns ``(output_tensor, loss_func_closure)`` as the schedule expects.
    """
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    batch = move_batch_to_cuda(batch)

    # Tag per-rank modality participation before the model forward so the
    # grad-finalization hook can correct encoder grads for partial DP
    # participation. ``mark_modality_participation`` ships in MM3 (#5286),
    # which is in-flight; guard the import so E3 stands alone and the call is a
    # no-op when the module is absent.
    # TODO(NMFW-516): make this import + call unconditional once MM3 lands.
    try:
        from examples.mimo.training.grad_sync import mark_modality_participation

        mark_modality_participation(model, batch)
    except ImportError:
        pass

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask=loss_mask)


def move_batch_to_cuda(value):
    """Move tensors in nested batch structures to the current CUDA device.

    Recurses through dicts, lists, and tuples, and through the tensor-valued
    fields of ``PackedSeqParams`` (which TE attention needs on the GPU). Leaves
    non-tensor leaves untouched.
    """
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch_to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_cuda(item) for item in value)
    # PackedSeqParams is a dataclass carrying tensors that TE attention needs
    # on the GPU. Recurse through its tensor-valued fields so cu_seqlens_q/kv
    # and max_seqlen_q/kv land on cuda alongside the rest of the batch.
    from megatron.core.packed_seq_params import PackedSeqParams

    if isinstance(value, PackedSeqParams):
        for attr in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
            "max_seqlen_q",
            "max_seqlen_kv",
        ):
            sub = getattr(value, attr, None)
            if isinstance(sub, torch.Tensor) and not sub.is_cuda:
                setattr(value, attr, sub.cuda(non_blocking=True))
        return value
    return value
