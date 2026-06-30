# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Simple mock data module for testing MIMO with image-text (VLM) models.

This module provides basic synthetic data generation for testing Vision Language Models
within the MIMO framework.
"""

from math import isqrt
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from megatron.core.packed_seq_params import PackedSeqParams


def create_mock_image(image_size: int = 336) -> torch.Tensor:
    """
    Create a simple mock image (all zeros).

    Args:
        image_size: Size of the square image

    Returns:
        Tensor of shape [3, H, W] with all zeros
    """
    return torch.zeros(3, image_size, image_size)


def create_mock_caption() -> str:
    """
    Create a simple mock caption.

    Returns:
        A simple caption string
    """
    return "This is an image."


def _dynamic_patch_grid(num_patches: int, require_even: bool) -> tuple[int, int]:
    """Factor a patch budget into the nearest-to-square valid grid."""
    for rows in range(isqrt(num_patches), 0, -1):
        if num_patches % rows:
            continue
        cols = num_patches // rows
        if require_even and (rows % 2 or cols % 2):
            continue
        return rows, cols
    qualifier = " even-by-even" if require_even else ""
    raise ValueError(f"cannot factor {num_patches} input patches into a{qualifier} patch grid")


class MockVLMDataset(Dataset):
    """Simple dataset of mock image-text pairs for VLM testing."""

    def __init__(
        self,
        size: int = 10000,
        image_size: int = 336,
        seq_len: int = 512,
        image_seq_length: int = 32,
        vocab_size: int = 256,
        tokenizer: Optional[Callable] = None,
        pad_token_id: int = 0,
        image_token_id: int = 32000,
        modality_module_name: str = "images",
        encoder_name: str = "clip_encoder",
        include_modality_inputs: bool = True,
        seed: int = 1234,
        dtype: torch.dtype = torch.float32,
        dynamic_resolution: bool = False,
        patch_dim: int = 16,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        pixel_shuffle: bool = False,
        num_image_tiles: int = 1,
        validate_image_token_count: bool = False,
    ):
        """
        Initialize the mock VLM dataset.

        Args:
            size: Number of examples in the dataset
            image_size: Size of the square images
            seq_len: Total length of the token sequence (image + text)
            image_seq_length: Number of image tokens to pad
            vocab_size: Size of the vocabulary for tokenization
            tokenizer: Optional tokenizer function
            pad_token_id: ID for padding token
            image_token_id: ID for image placeholder token
            validate_image_token_count: Enforce RADIO image-token/geometry parity
        """
        self.size = size
        self.image_size = image_size
        self.seq_len = seq_len
        self.image_seq_length = image_seq_length
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.modality_module_name = modality_module_name
        self.encoder_name = encoder_name
        self.include_modality_inputs = include_modality_inputs
        self.seed = seed
        self.dtype = dtype
        self.dynamic_resolution = dynamic_resolution
        self.patch_dim = patch_dim
        self.img_h = image_size if img_h is None else img_h
        self.img_w = image_size if img_w is None else img_w
        self.pixel_shuffle = pixel_shuffle
        self.num_image_tiles = num_image_tiles
        self.validate_image_token_count = validate_image_token_count

        # Special token IDs
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self._text_token_ids = torch.arange(1, self.vocab_size, dtype=torch.long)
        self._text_token_ids = self._text_token_ids[
            (self._text_token_ids != self.image_token_id)
            & (self._text_token_ids != self.pad_token_id)
        ]

        if self.seq_len <= self.image_seq_length:
            raise ValueError(
                f"image_seq_length ({self.image_seq_length}) must be less than "
                f"seq_len ({self.seq_len})"
            )
        if self.patch_dim <= 0:
            raise ValueError(f"patch_dim must be positive, got {self.patch_dim}")
        if self.num_image_tiles <= 0:
            raise ValueError(f"num_image_tiles must be positive, got {self.num_image_tiles}")

        if self.dynamic_resolution:
            if self.image_seq_length % self.num_image_tiles:
                raise ValueError(
                    f"image_seq_length ({self.image_seq_length}) must be divisible by "
                    f"num_image_tiles ({self.num_image_tiles})"
                )
            emitted_per_tile = self.image_seq_length // self.num_image_tiles
            patches_per_tile = emitted_per_tile * (4 if self.pixel_shuffle else 1)
            self.patch_rows, self.patch_cols = _dynamic_patch_grid(
                patches_per_tile, require_even=self.pixel_shuffle
            )
        else:
            if self.img_h % self.patch_dim or self.img_w % self.patch_dim:
                raise ValueError(
                    f"img_h ({self.img_h}) and img_w ({self.img_w}) must be divisible by "
                    f"patch_dim ({self.patch_dim})"
                )
            self.patch_rows = self.img_h // self.patch_dim
            self.patch_cols = self.img_w // self.patch_dim
        if self.validate_image_token_count and not self.dynamic_resolution:
            if self.pixel_shuffle and self.patch_rows != self.patch_cols:
                raise ValueError(
                    "fixed-resolution RADIO pixel shuffle requires a square patch grid, "
                    f"got {self.patch_rows}x{self.patch_cols}"
                )
            if self.pixel_shuffle and (self.patch_rows % 2 or self.patch_cols % 2):
                raise ValueError(
                    "pixel shuffle requires an even patch grid in both dimensions, "
                    f"got {self.patch_rows}x{self.patch_cols}"
                )
            patches = self.num_image_tiles * self.patch_rows * self.patch_cols
            emitted_tokens = patches // 4 if self.pixel_shuffle else patches
            if self.image_seq_length != emitted_tokens:
                raise ValueError(
                    f"fixed-resolution mode emits {emitted_tokens} image tokens, "
                    f"got image_seq_length={self.image_seq_length}"
                )

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            idx: Index used to deterministically seed this sample's text tokens.

        Returns:
            Dictionary containing:
            - input_ids: Tokenized caption with image token
            - labels: Shifted input_ids for language modeling
            - loss_mask: Mask for loss calculation
            - position_ids: Position IDs for the tokens
            - modality_inputs: Nested modality/encoder inputs, or an empty dictionary
        """
        # Generate random token sequence for this sample.
        input_ids = self._mock_tokenize(idx)

        # Create labels (shifted input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:]
        labels[labels == self.image_token_id] = -100

        # Create loss mask (1 for tokens to calculate loss on, 0 for others)
        loss_mask = (labels != -100).float()

        # Create position IDs (just sequential integers)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "modality_inputs": {},
        }
        if self.include_modality_inputs:
            sample["modality_inputs"] = {
                self.modality_module_name: {self.encoder_name: self._encoder_inputs()}
            }
        return sample

    def _mock_tokenize(self, idx: int) -> torch.Tensor:
        """
        Generate a mock token sequence consisting of ``image_seq_length`` image tokens followed by
        randomly generated text tokens such that the total sequence length equals
        ``self.seq_len``.

        Returns:
            torch.Tensor: Tensor of token IDs of shape ``[seq_len]``.
        """

        # Image placeholder tokens ─ placed at the beginning of the sequence to mimic
        # the layout produced by many VLM tokenizers.
        image_tokens = torch.full(
            (self.image_seq_length,), self.image_token_id, dtype=torch.long
        )

        # Random text tokens excluding the configured special IDs.
        num_text_tokens = self.seq_len - self.image_seq_length
        if num_text_tokens and self._text_token_ids.numel() == 0:
            raise ValueError(
                "vocab_size must contain at least one non-padding token distinct from "
                "image_token_id"
            )
        if num_text_tokens:
            generator = torch.Generator().manual_seed(self.seed + idx)
            choices = torch.randint(
                self._text_token_ids.numel(),
                (num_text_tokens,),
                generator=generator,
                dtype=torch.long,
            )
            text_tokens = self._text_token_ids[choices]
        else:
            text_tokens = torch.empty(0, dtype=torch.long)

        # Concatenate to form the full sequence.
        token_ids = torch.cat((image_tokens, text_tokens), dim=0)

        return token_ids

    def _encoder_inputs(self) -> Dict[str, torch.Tensor]:
        """Build one sample's fixed- or dynamic-resolution encoder inputs on CPU."""
        if not self.dynamic_resolution:
            return {
                "x": torch.zeros(
                    self.num_image_tiles, 3, self.img_h, self.img_w, dtype=self.dtype
                )
            }

        patches_per_tile = self.patch_rows * self.patch_cols
        return {
            "x": torch.zeros(
                1,
                self.num_image_tiles * patches_per_tile,
                3 * self.patch_dim**2,
                dtype=self.dtype,
            ),
            "imgs_sizes": torch.tensor(
                [
                    [self.patch_rows * self.patch_dim, self.patch_cols * self.patch_dim]
                ]
                * self.num_image_tiles,
                dtype=torch.int32,
            ),
        }


def get_mock_vlm_dataloader(
    batch_size: int = 8,
    dataset_size: int = 100,
    image_size: int = 224,
    seq_len: int = 77,
    image_seq_length: int = 32,
    num_workers: int = 0,
    pad_token_id: int = 0,
    image_token_id: int = 50000,
    vocab_size: int = 256,
    modality_module_name: str = "images",
    encoder_name: str = "clip_encoder",
    include_modality_inputs: bool = True,
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
    dynamic_resolution: bool = False,
    patch_dim: int = 16,
    img_h: Optional[int] = None,
    img_w: Optional[int] = None,
    pixel_shuffle: bool = False,
    num_image_tiles: int = 1,
    shuffle: bool = True,
    validate_image_token_count: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for mock VLM data.

    Args:
        batch_size: Batch size
        dataset_size: Size of the dataset
        image_size: Size of the square images
        seq_len: Total length of the token sequence (image + text)
        image_seq_length: Number of image tokens to pad
        num_workers: Number of worker processes for data loading
        pad_token_id: ID for padding token
        image_token_id: ID for image placeholder token

    Returns:
        DataLoader for the mock VLM dataset
    """
    dataset = MockVLMDataset(
        size=dataset_size,
        image_size=image_size,
        seq_len=seq_len,
        image_seq_length=image_seq_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
        modality_module_name=modality_module_name,
        encoder_name=encoder_name,
        include_modality_inputs=include_modality_inputs,
        seed=seed,
        dtype=dtype,
        dynamic_resolution=dynamic_resolution,
        patch_dim=patch_dim,
        img_h=img_h,
        img_w=img_w,
        pixel_shuffle=pixel_shuffle,
        num_image_tiles=num_image_tiles,
        validate_image_token_count=validate_image_token_count,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    return dataloader


def _collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """
    Collate function for the DataLoader.

    Args:
        batch: List of dictionaries from the dataset

    Returns:
        Dictionary of batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])

    collated = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {},
    }
    for modality_name, encoders in batch[0]["modality_inputs"].items():
        collated["modality_inputs"][modality_name] = {}
        for encoder_name in encoders:
            encoder_items = [
                item["modality_inputs"][modality_name][encoder_name] for item in batch
            ]
            x = encoder_items[0]["x"]
            encoder_batch = {
                "x": torch.cat([item["x"] for item in encoder_items], dim=1 if x.ndim == 3 else 0)
            }
            if "imgs_sizes" in encoder_items[0]:
                imgs_sizes = torch.cat([item["imgs_sizes"] for item in encoder_items])
                patch_area = x.shape[-1] // 3
                patch_dim = isqrt(patch_area)
                if 3 * patch_dim**2 != x.shape[-1]:
                    raise ValueError(
                        f"dynamic encoder feature size ({x.shape[-1]}) is not 3 * patch_dim^2"
                    )
                seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1, dtype=torch.int32)
                cu_seqlens = torch.cat(
                    (torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0))
                )
                max_seqlen = int(seq_lens.max().item())
                encoder_batch.update(
                    {
                        "imgs_sizes": imgs_sizes,
                        "packed_seq_params": PackedSeqParams(
                            qkv_format="thd",
                            cu_seqlens_q=cu_seqlens,
                            cu_seqlens_kv=cu_seqlens.clone(),
                            max_seqlen_q=max_seqlen,
                            max_seqlen_kv=max_seqlen,
                        ),
                    }
                )
            collated["modality_inputs"][modality_name][encoder_name] = encoder_batch
    return collated


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide datasets for training, validation, and testing."""
    from megatron.core import mpu
    from megatron.training import get_args

    args = get_args()

    # Print some info to confirm args are available
    print(f"Creating datasets with batch size: {args.micro_batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Image sequence length: {args.image_seq_length}")
    print(f"Total sequence length: {args.total_seq_length}")

    # Only build dataset on tensor parallel rank 0
    if mpu.get_tensor_model_parallel_rank() == 0:

        from examples.mimo.data.mock import MockVLMDataset

        train_dataset = MockVLMDataset(
            size=train_val_test_num_samples[0],
            image_size=args.image_size,
            seq_len=args.total_seq_length,
            image_seq_length=args.image_seq_length,
            pad_token_id=args.pad_token_id,
            image_token_id=args.image_token_id,
        )

        # Use the same dataset type for validation
        valid_dataset = MockVLMDataset(
            size=train_val_test_num_samples[1] if train_val_test_num_samples[1] > 0 else 100,
            image_size=args.image_size,
            seq_len=args.total_seq_length,
            image_seq_length=args.image_seq_length,
            pad_token_id=args.pad_token_id,
            image_token_id=args.image_token_id,
        )

        # No test dataset for now
        test_dataset = None
    else:
        train_dataset = None
        valid_dataset = None
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    print("\nCreating mock VLM dataloader...")
    dataloader = get_mock_vlm_dataloader(batch_size=4, dataset_size=10)

    print(f"DataLoader has {len(dataloader)} batches")

    for batch in dataloader:
        print("\nBatch from dataloader:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        break
