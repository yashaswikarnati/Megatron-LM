# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-only synthetic VLM data iterator for MIMO throughput benchmarking.

Generates batches directly on GPU with no DataLoader overhead, matching the
format expected by MimoModel's forward pass: input_ids with image tokens,
labels with -100 masking, loss_mask, position_ids, and nested modality_inputs.
"""

import torch


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
    ):
        self.encoder_hidden_size = encoder_hidden_size
        self.image_seq_length = image_seq_length
        self.total_seq_length = total_seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.encoder_name = encoder_name

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Generate a batch directly on GPU.

        Returns:
            Dict with input_ids, labels, loss_mask, position_ids, and modality_inputs.
        """
        # Encoder hidden states: [image_seq_length, micro_batch_size, encoder_hidden_size]
        encoder_hidden_states = torch.randn(
            self.image_seq_length,
            self.micro_batch_size,
            self.encoder_hidden_size,
            device='cuda',
            dtype=torch.bfloat16,
        )

        # Input IDs: first image_seq_length tokens are image_token_id, rest are random vocab
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            self.image_token_id,
            dtype=torch.long,
            device='cuda',
        )
        # Text tokens in [1, vocab_size) with image_token_id excluded
        upper = min(self.image_token_id, self.vocab_size)
        text_tokens = torch.randint(
            1,
            upper,
            (self.micro_batch_size, self.total_seq_length - self.image_seq_length),
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

        # Modality inputs: nested dict matching VisionModalitySubmodules expectations
        modality_inputs = {
            self.encoder_name: {
                "clip_encoder": {
                    'hidden_states': encoder_hidden_states,
                    'attention_mask': None,
                }
            }
        }

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "modality_inputs": modality_inputs,
        }
