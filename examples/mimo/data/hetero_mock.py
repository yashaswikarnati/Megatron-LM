# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mock VLM data provider for heterogeneous MIMO training examples."""

from __future__ import annotations

import argparse

import torch

from examples.mimo.utils.hetero import debug_rank


def validate_mock_data_args(args: argparse.Namespace) -> None:
    """Validate synthetic next-token VLM data constraints."""
    image_seq_length = args.image_seq_length or args.seq_length // 2
    if image_seq_length >= args.seq_length:
        raise ValueError("--image-seq-length must be smaller than --seq-length")
    if args.seq_length - image_seq_length < 2:
        raise ValueError("mock next-token training needs at least two text tokens")


class MockVLMIterator:
    """Infinite iterator yielding synthetic VLM-like microbatches."""

    def __init__(
        self, args: argparse.Namespace, micro_batch_size: int, encoder_name: str, seed: int
    ) -> None:
        self.args = args
        self.micro_batch_size = micro_batch_size
        self.encoder_name = encoder_name
        self.image_seq_length = args.image_seq_length or args.seq_length // 2
        self.vision_encoder_key = getattr(args, "vision_encoder_key", "clip_encoder")
        self.vision_input_mode = getattr(args, "vision_input_mode", "hidden_states")
        self.dtype = torch.float32 if args.fp32 else torch.bfloat16
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(seed)
        if self.image_seq_length >= args.seq_length:
            raise ValueError("--image-seq-length must be smaller than --seq-length")

    def __iter__(self):
        return self

    def __next__(self):
        args = self.args
        debug_rank(
            f"mock batch start: micro_batch_size={self.micro_batch_size}, "
            f"image_seq_length={self.image_seq_length}"
        )
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            args.image_token_id,
            dtype=torch.long,
            device="cuda",
        )
        text_tokens = torch.randint(
            1,
            args.vocab_size,
            (self.micro_batch_size, args.seq_length - self.image_seq_length),
            device="cuda",
            generator=self.generator,
        )
        special_token_ids = {args.image_token_id, args.pad_token_id}
        replacement_token_id = next(
            (
                token_id
                for token_id in range(1, args.vocab_size)
                if token_id not in special_token_ids
            ),
            None,
        )
        if replacement_token_id is None:
            raise RuntimeError("mock data needs at least one non-special token id")
        if 1 <= args.image_token_id < args.vocab_size:
            text_tokens[text_tokens == args.image_token_id] = replacement_token_id
        if 1 <= args.pad_token_id < args.vocab_size:
            text_tokens[text_tokens == args.pad_token_id] = replacement_token_id
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        labels[(labels == args.image_token_id) | (labels == args.pad_token_id)] = -100
        loss_mask = (labels != -100).to(dtype=torch.float32)

        if self.vision_input_mode == "pixels":
            encoder_inputs = {
                self.vision_encoder_key: {
                    "x": torch.randn(
                        self.micro_batch_size * args.num_image_tiles,
                        3,
                        args.img_h,
                        args.img_w,
                        device="cuda",
                        dtype=self.dtype,
                        generator=self.generator,
                    )
                }
            }
        else:
            encoder_hidden_states = torch.randn(
                self.image_seq_length,
                self.micro_batch_size,
                args.hidden_size,
                device="cuda",
                dtype=self.dtype,
                generator=self.generator,
            )
            encoder_inputs = {
                self.vision_encoder_key: {
                    "hidden_states": encoder_hidden_states,
                    "attention_mask": None,
                }
            }

        num_image_placeholders = (input_ids == args.image_token_id).sum().item()
        expected_image_placeholders = self.image_seq_length * self.micro_batch_size
        if num_image_placeholders != expected_image_placeholders:
            raise RuntimeError(
                f"mock batch has {num_image_placeholders} image placeholders, "
                f"expected {expected_image_placeholders}"
            )

        debug_rank("mock batch ready")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(args.seq_length, device="cuda")
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone(),
            "modality_inputs": {self.encoder_name: {**encoder_inputs}},
        }
