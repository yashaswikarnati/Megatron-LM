# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the heterogeneous MIMO mock-data path."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.data.mock import MockVLMDataset, get_mock_vlm_dataloader
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from megatron.core.packed_seq_params import PackedSeqParams


def _group(rank=0, size=1):
    return SimpleNamespace(rank=lambda: rank, size=lambda: size)


def _grid(contains_rank):
    return SimpleNamespace(is_current_rank_in_grid=lambda: contains_rank)


def _args(**overrides):
    values = dict(
        seed=123,
        dataset_provider="mock",
        micro_batch_size=2,
        llm_dp=2,
        encoder_dp=1,
        seq_length=8,
        image_seq_length=4,
        vocab_size=64,
        image_token_id=63,
        params_dtype=torch.float32,
        dynamic_resolution=False,
        patch_dim=2,
        img_h=4,
        img_w=4,
        pixel_shuffle=False,
        num_image_tiles=1,
        mock_dataset_size=16,
        disable_vision_class_token=True,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _topology(*, language_rank, encoder_rank=None, language_pp_rank=0):
    encoder = RADIO_ENCODER_MODULE_NAME
    grids = {"language": _grid(language_rank)}
    pgs = {
        "language": SimpleNamespace(
            pp=_group(rank=language_pp_rank, size=3), dp=_group(rank=0, size=2)
        )
    }
    if encoder_rank is not None:
        grids[encoder] = _grid(encoder_rank)
        pgs[encoder] = SimpleNamespace(pp=_group(), dp=_group(rank=1, size=2))
    return SimpleNamespace(grids=grids, module_pgs=pgs)


@pytest.fixture
def adapter(monkeypatch):
    from examples.mimo.training import data

    monkeypatch.setattr(data, "get_pg_rank", lambda pg: pg.rank())
    monkeypatch.setattr(data, "is_pp_first_stage", lambda pg: pg.rank() == 0)
    monkeypatch.setattr(data, "is_pp_last_stage", lambda pg: pg.rank() == pg.size() - 1)
    return data


def test_mock_loader_preserves_nested_inputs_and_masks_shifted_labels():
    loader = get_mock_vlm_dataloader(
        batch_size=2,
        dataset_size=2,
        shuffle=False,
        seq_len=16,
        image_seq_length=12,
        vocab_size=32,
        pad_token_id=5,
        image_token_id=7,
        modality_module_name="images",
        encoder_name="clip_encoder",
        img_h=8,
        img_w=12,
        patch_dim=4,
        num_image_tiles=2,
        dtype=torch.float16,
        seed=11,
        validate_image_token_count=True,
    )

    batch = next(iter(loader))
    encoder_inputs = batch["modality_inputs"]["images"]["clip_encoder"]
    assert encoder_inputs["x"].shape == (4, 3, 8, 12)
    assert encoder_inputs["x"].dtype == torch.float16
    assert torch.all(batch["input_ids"][:, :12] == 7)
    assert torch.all(batch["input_ids"][:, 12:] != 7)
    assert torch.all(batch["input_ids"][:, 12:] != 5)

    expected_labels = torch.cat(
        (batch["input_ids"][:, 1:], torch.full((2, 1), -100, dtype=torch.long)), dim=1
    )
    expected_labels[expected_labels == 7] = -100
    assert torch.equal(batch["labels"], expected_labels)
    assert torch.equal(batch["loss_mask"], (expected_labels != -100).float())
    assert len(MockVLMDataset(image_size=224, seq_len=512, image_seq_length=197)) == 10_000
    legacy_loader = get_mock_vlm_dataloader(batch_size=1, dataset_size=1)
    assert (legacy_loader.dataset.seq_len, legacy_loader.dataset.image_seq_length) == (77, 32)


def test_dynamic_radio_loader_emits_patchified_cpu_metadata():
    loader = get_mock_vlm_dataloader(
        batch_size=2,
        dataset_size=2,
        shuffle=False,
        seq_len=24,
        image_seq_length=12,
        vocab_size=64,
        image_token_id=63,
        modality_module_name=RADIO_ENCODER_MODULE_NAME,
        encoder_name=RADIO_ENCODER_MODULE_NAME,
        dynamic_resolution=True,
        pixel_shuffle=True,
        patch_dim=8,
        img_h=224,
        img_w=224,
        num_image_tiles=3,
        dtype=torch.bfloat16,
        validate_image_token_count=True,
    )

    inputs = next(iter(loader))["modality_inputs"][RADIO_ENCODER_MODULE_NAME][
        RADIO_ENCODER_MODULE_NAME
    ]
    assert inputs["x"].shape == (1, 96, 3 * 8 * 8)
    assert inputs["x"].dtype == torch.bfloat16
    assert inputs["imgs_sizes"].shape == (6, 2)
    assert inputs["imgs_sizes"].dtype == torch.int32
    assert inputs["imgs_sizes"].device.type == "cpu"
    assert torch.equal(inputs["imgs_sizes"], torch.full((6, 2), 32, dtype=torch.int32))

    packed = inputs["packed_seq_params"]
    assert isinstance(packed, PackedSeqParams)
    assert (packed.qkv_format, packed.max_seqlen_q, packed.max_seqlen_kv) == ("thd", 16, 16)
    assert torch.equal(packed.cu_seqlens_q, torch.arange(0, 97, 16, dtype=torch.int32))
    assert torch.equal(packed.cu_seqlens_kv, packed.cu_seqlens_q)
    assert packed.cu_seqlens_q.device.type == "cpu"

    with pytest.raises(ValueError, match="image_seq_length.*divisible by num_image_tiles"):
        MockVLMDataset(
            dynamic_resolution=True, pixel_shuffle=True, image_seq_length=10, num_image_tiles=3
        )
    with pytest.raises(ValueError, match="must be less than seq_len"):
        MockVLMDataset(seq_len=4, image_seq_length=4)
    with pytest.raises(ValueError, match="pixel shuffle.*even patch grid"):
        MockVLMDataset(
            pixel_shuffle=True,
            patch_dim=8,
            img_h=24,
            img_w=32,
            image_seq_length=3,
            validate_image_token_count=True,
        )
    with pytest.raises(ValueError, match="fixed-resolution.*12 image tokens"):
        MockVLMDataset(
            seq_len=16, image_seq_length=4, patch_dim=4, img_h=8, img_w=12,
            num_image_tiles=2, validate_image_token_count=True,
        )
    with pytest.raises(ValueError, match="square patch grid"):
        MockVLMDataset(
            seq_len=8, image_seq_length=2, patch_dim=4, img_h=8, img_w=16,
            pixel_shuffle=True, validate_image_token_count=True,
        )


def test_data_adapter_builds_independent_role_specific_loaders(adapter):
    language_loaders = adapter.build_train_valid_test_data_loaders(
        _args(), _topology(language_rank=True)
    )
    assert all(loader.batch_size == 2 for loader in language_loaders)
    assert len({id(loader.dataset) for loader in language_loaders}) == 3
    assert len({loader.dataset.seed for loader in language_loaders}) == 3
    language_batch = next(iter(language_loaders[0]))
    assert language_batch["input_ids"].shape == (2, 8)
    assert language_batch["modality_inputs"] == {}
    assert all(
        adapter.build_train_valid_test_data_loaders(
            _args(), _topology(language_rank=True, language_pp_rank=2)
        )
    )

    encoder_loaders = adapter.build_train_valid_test_data_loaders(
        _args(), _topology(encoder_rank=True, language_rank=False)
    )
    assert all(loader.batch_size == 4 for loader in encoder_loaders)
    assert encoder_loaders[0].dataset.validate_image_token_count
    encoder_batch = next(iter(encoder_loaders[0]))
    assert encoder_batch["input_ids"].shape == (4, 8)
    encoder_inputs = encoder_batch["modality_inputs"][RADIO_ENCODER_MODULE_NAME][
        RADIO_ENCODER_MODULE_NAME
    ]
    assert encoder_inputs["x"].shape == (4, 3, 4, 4)
    fixed = adapter._mock_loader_kwargs(
        _args(image_seq_length=None, seq_length=16, img_w=6, num_image_tiles=2),
        RADIO_ENCODER_MODULE_NAME,
    )
    dynamic = adapter._mock_loader_kwargs(
        _args(
            image_seq_length=None,
            seq_length=24,
            dynamic_resolution=True,
            pixel_shuffle=True,
            num_image_tiles=3,
        ),
        RADIO_ENCODER_MODULE_NAME,
    )
    assert (fixed["image_seq_length"], dynamic["image_seq_length"]) == (12, 12)


def test_data_adapter_rejects_invalid_encoder_inputs_and_skips_non_consumers(adapter):
    loaders = adapter.build_train_valid_test_data_loaders(
        _args(), _topology(encoder_rank=False, language_rank=True, language_pp_rank=1)
    )
    assert loaders == (None, None, None)
    assert all(
        adapter.build_train_valid_test_data_loaders(
            _args(disable_vision_class_token=False),
            _topology(encoder_rank=False, language_rank=True),
        )
    )

    with pytest.raises(ValueError, match="micro_batch_size.*llm_dp.*encoder_dp"):
        adapter.build_train_valid_test_data_loaders(
            _args(micro_batch_size=1, llm_dp=1, encoder_dp=2),
            _topology(encoder_rank=False, language_rank=True),
        )
    with pytest.raises(ValueError, match="disable-vision-class-token"):
        adapter.build_train_valid_test_data_loaders(
            _args(disable_vision_class_token=False),
            _topology(encoder_rank=True, language_rank=False),
        )
