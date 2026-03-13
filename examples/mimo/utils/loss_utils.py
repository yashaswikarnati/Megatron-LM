# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Loss functions for MIMO model training."""

import math

import torch

from megatron.core.parallel_state import get_context_parallel_group, get_data_parallel_group
from megatron.training import get_args


def _get_mask_start_and_end_idx(arr):
    """Find contiguous non-zero spans in a 1-D tensor.

    Returns a list of (start, end) tuples (inclusive) for each span.
    E.g. arr = [0, 1, 0, 0, 1, 1] → [(1, 1), (4, 5)]
    """
    if len(arr) == 0:
        return []
    mask_int = (arr != 0).int()
    diff = mask_int[1:] - mask_int[:-1]
    start_indices = (diff == 1).nonzero(as_tuple=False).flatten() + 1
    end_indices = (diff == -1).nonzero(as_tuple=False).flatten()
    if mask_int[0]:
        start_indices = torch.cat((torch.tensor([0], device=arr.device), start_indices))
    if mask_int[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(arr) - 1], device=arr.device)))
    return list(zip(start_indices.tolist(), end_indices.tolist()))


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    args = get_args()
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)

    loss = torch.cat([total_loss.view(1), total_tokens.view(1)])

    loss_for_backward = loss[0].clone()
    # If CP is active, reduce the loss across all CP ranks
    # as they have loss calculated for their own sequence shards.
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=get_context_parallel_group())
        loss_for_backward = loss[0].clone()
    # For reporting, clone and detach the loss.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    return (loss_for_backward, local_num_tokens, {'lm loss': (reporting_loss)})


def scaled_loss_func(loss_mask, output_tensor):
    """Per-conversation-turn sqrt-weighted loss.

    Scale the loss so that each conversation turn contributes proportional
    to sqrt(n_tokens) rather than n_tokens, preventing long turns from
    dominating short ones.

    Formula:
        L = (1 / Σ_j √n_j) × Σ_i [ (Σ_t loss_t_i / n_i) × √n_i ]
    """
    losses = output_tensor.float()

    loss_list = []
    num_valid_labels_list = []
    for idx in range(losses.shape[0]):
        loss_this_sample = losses[idx]
        turn_spans = _get_mask_start_and_end_idx(loss_mask[idx])
        for turn_start, turn_end in turn_spans:
            n_tokens = turn_end - turn_start + 1
            turn_loss = loss_this_sample[turn_start : turn_end + 1].sum() / n_tokens
            loss_list.append(turn_loss)
            num_valid_labels_list.append(n_tokens)

    if len(loss_list) == 0:
        raise RuntimeError(
            "scaled_loss_func: no valid conversation turns found in batch "
            "(loss_mask is all zeros)"
        )

    base_num = sum(math.sqrt(n) for n in num_valid_labels_list)
    for i in range(len(loss_list)):
        loss_list[i] = loss_list[i] * math.sqrt(num_valid_labels_list[i]) / base_num

    total_loss = torch.stack(loss_list).sum()
    num_tokens = torch.ones_like(total_loss).to(torch.int)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), num_tokens.view(1)])

    return (total_loss, num_tokens, {'lm loss': reporting_loss})
