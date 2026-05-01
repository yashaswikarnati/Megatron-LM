# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-only synthetic VLM data iterator for MIMO throughput benchmarking.

Generates batches directly on GPU with no DataLoader overhead, matching the
format expected by MimoModel's forward pass: input_ids with image tokens,
labels with -100 masking, loss_mask, position_ids, and nested modality_inputs.
"""

import torch


def get_non_colocated_data_ownership(
    *,
    in_encoder_grid: bool,
    encoder_pp_rank: int | None,
    in_llm_grid: bool,
    llm_pp_rank: int | None,
    llm_pp_size: int | None,
) -> tuple[bool, bool]:
    """Return ``(needs_text, needs_encoder)`` for a non-colocated rank."""
    needs_encoder = in_encoder_grid and encoder_pp_rank == 0
    needs_text = (
        in_llm_grid
        and llm_pp_rank is not None
        and llm_pp_size is not None
        and (llm_pp_rank == 0 or llm_pp_rank == llm_pp_size - 1)
    )
    return needs_text, needs_encoder


def compute_non_colocated_encoder_batch_size(
    micro_batch_size: int,
    num_images_per_sample: int,
    encoder_dp: int,
    llm_dp: int,
) -> int:
    """Return per-encoder-DP image batch size for non-colocated bridge traffic."""
    total_images_per_llm_microbatch = micro_batch_size * num_images_per_sample * llm_dp
    if total_images_per_llm_microbatch % encoder_dp != 0:
        raise ValueError(
            "Total encoder images per LLM microbatch "
            f"({micro_batch_size} * {num_images_per_sample} * {llm_dp} = "
            f"{total_images_per_llm_microbatch}) must be divisible by encoder DP "
            f"({encoder_dp})"
        )
    return total_images_per_llm_microbatch // encoder_dp


class SyntheticVLMIterator:
    """GPU-only synthetic data for VLM throughput benchmarking.

    Generates batches with:
    - input_ids: [batch, seq_len] with image_token_id for first image_seq_length tokens
    - labels: same as input_ids but -100 for image tokens
    - loss_mask: 1.0 for text tokens, 0.0 for image tokens
    - position_ids: [batch, seq_len] sequential
    - modality_inputs: {"<encoder_name>": {"clip_encoder": {"hidden_states": tensor, "attention_mask": None}}}
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        image_seq_length: int,
        total_seq_length: int,
        micro_batch_size: int,
        vocab_size: int,
        image_token_id: int = 32000,
        encoder_name: str = "images",
        num_images_per_sample: int = 1,
        include_text: bool = True,
        include_encoder: bool = True,
        encoder_batch_size: int | None = None,
    ):
        self.encoder_hidden_size = encoder_hidden_size
        self.image_seq_length = image_seq_length
        self.total_seq_length = total_seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.encoder_name = encoder_name
        self.num_images_per_sample = num_images_per_sample
        self.total_image_tokens = num_images_per_sample * image_seq_length
        self.include_text = include_text
        self.include_encoder = include_encoder
        self.encoder_batch_size = encoder_batch_size

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Generate a batch directly on GPU.

        Returns:
            Dict with input_ids, labels, loss_mask, position_ids, and modality_inputs.
        """
        batch = {
            "input_ids": None,
            "labels": None,
            "loss_mask": None,
            "position_ids": None,
            "modality_inputs": None,
        }

        if self.include_encoder:
            # Encoder hidden states: [image_seq_length, image_batch, hidden].
            # In colocated mode image_batch is num_images * text micro-batch.
            # In non-colocated mode it can be set directly to the per-encoder-DP
            # image count required by the bridge fan-in/fan-out plan.
            encoder_batch = (
                self.encoder_batch_size
                if self.encoder_batch_size is not None
                else self.num_images_per_sample * self.micro_batch_size
            )
            encoder_hidden_states = torch.randn(
                self.image_seq_length,
                encoder_batch,
                self.encoder_hidden_size,
                device='cuda',
                dtype=torch.bfloat16,
            )
            batch["modality_inputs"] = {
                self.encoder_name: {
                    "clip_encoder": {
                        'hidden_states': encoder_hidden_states,
                        'attention_mask': None,
                    }
                }
            }

        if self.include_text:
            # Input IDs: first total_image_tokens are image_token_id, rest are text
            image_tokens = torch.full(
                (self.micro_batch_size, self.total_image_tokens),
                self.image_token_id,
                dtype=torch.long,
                device='cuda',
            )
            # Text tokens in [1, vocab_size) with image_token_id excluded
            upper = min(self.image_token_id, self.vocab_size)
            text_len = self.total_seq_length - self.total_image_tokens
            text_tokens = torch.randint(
                1,
                upper,
                (self.micro_batch_size, text_len),
                dtype=torch.long,
                device='cuda',
            )
            input_ids = torch.cat([image_tokens, text_tokens], dim=1)

            # Labels: clone of input_ids, -100 where image tokens
            labels = input_ids.clone()
            labels[input_ids == self.image_token_id] = -100

            # Loss mask: 0.0 for image tokens, 1.0 for text tokens
            loss_mask = torch.ones(
                self.micro_batch_size,
                self.total_seq_length,
                device='cuda',
                dtype=torch.float32,
            )
            loss_mask[input_ids == self.image_token_id] = 0.0

            # Position IDs: sequential [0, 1, ..., total_seq_length-1] expanded to batch
            position_ids = (
                torch.arange(self.total_seq_length, device='cuda')
                .unsqueeze(0)
                .expand(self.micro_batch_size, -1)
                .clone()
            )

            batch.update(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "position_ids": position_ids,
                }
            )

        return batch
