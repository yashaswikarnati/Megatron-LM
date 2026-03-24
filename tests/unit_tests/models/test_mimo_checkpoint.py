# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for MIMO model distributed checkpoint save/load in non-colocated mode.

Run from the worktree directory:
    cd <worktree> && uv run python -m torch.distributed.run --nproc-per-node=2 tests/unit_tests/models/test_mimo_checkpoint.py
"""

# Ensure worktree megatron takes precedence when multiple copies are on sys.path
import os as _os, sys as _sys  # noqa: E402

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
if _root in _sys.path:
    _sys.path.remove(_root)
_sys.path.insert(0, _root)


import logging
import os
import shutil
import tempfile

import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Grid and Process Group Helpers
# ============================================================================


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with specified parallelism."""
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["ep"])
    # Extra groups needed by optimizer
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    return grid


def get_pg_collection(grid):
    """Get ProcessGroupCollection from grid."""
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    pg_collection.dp = grid.get_pg("dp")
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    return pg_collection


def create_all_embedding_groups(*grids):
    """Create embedding groups for all grids collectively.

    dist.new_group() is a world-collective: ALL ranks must call it in the
    same order, even if they're not members. This function enumerates all
    PP pipelines across all grids and creates the pos_embd and embd groups
    for each in a deterministic order that all ranks follow.

    Returns a dict mapping grid id -> list of (pos_embd_pg, embd_pg, pp_ranks).
    """
    grid_embedding_info = {}
    for grid in grids:
        pp_enum = grid.get_rank_enum("pp")
        pipeline_groups = []
        for pp_ranks in pp_enum:
            pp_ranks = sorted(pp_ranks)
            pos_embd_ranks = [pp_ranks[0]]
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
            embd_pg = dist.new_group(ranks=embd_ranks)
            pipeline_groups.append((pos_embd_pg, embd_pg, pp_ranks))
        grid_embedding_info[id(grid)] = pipeline_groups
    return grid_embedding_info


def get_pg_collection_with_embedding_groups(grid, embedding_info):
    """Get ProcessGroupCollection with pre-created embedding groups."""
    from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

    pg_collection = get_pg_collection(grid)

    if not pg_collection.pp:
        pg_collection.pos_embd = None
        pg_collection.embd = None
        return pg_collection

    my_pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
    for pos_embd_pg, embd_pg, pp_ranks in embedding_info[id(grid)]:
        if pp_ranks == my_pp_ranks:
            pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
            pg_collection.embd = (
                embd_pg
                if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
                else None
            )
            return pg_collection

    pg_collection.pos_embd = None
    pg_collection.embd = None
    return pg_collection


def is_rank_in_grid(grid):
    """Check if current rank is in grid."""
    rank = dist.get_rank()
    return grid.rank_offset <= rank < grid.rank_offset + grid.size


# ============================================================================
# Model Spec Helpers
# ============================================================================


def get_language_model_spec(
    num_layers, hidden_size, num_attention_heads, vocab_size, seq_len, pg_collection
):
    """Get the language model spec."""
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
    )
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
            "pg_collection": pg_collection,
        },
    )


def get_vision_submodules_spec(
    num_layers, hidden_size, num_attention_heads, language_hidden_size, pg_collection
):
    """Get the submodule spec for the vision modality."""
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1
    pp_rank = dist.get_rank(pg_collection.pp)

    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    vision_encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": get_gpt_layer_with_transformer_engine_spec(),
            "pg_collection": pg_collection,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
        },
    )

    proj_config = TransformerConfig(
        num_layers=1, hidden_size=language_hidden_size, num_attention_heads=1
    )
    proj_config.ffn_hidden_size = language_hidden_size
    proj_config.bias_activation_fusion = True
    proj_config.add_bias_linear = True
    proj_config.activation_func = torch.nn.functional.gelu

    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": proj_config,
            "submodules": ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
            ).submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "pg_collection": pg_collection,
        },
    )

    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )


# ============================================================================
# Model Creation
# ============================================================================


def create_mimo_model(
    encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seq_len, seed, embedding_info
):
    """Create a MiMo model with deterministic weights for checkpoint testing."""
    language_pg = get_pg_collection_with_embedding_groups(llm_grid, embedding_info)
    vision_pg = get_pg_collection_with_embedding_groups(encoder_grid, embedding_info)

    language_model_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pg_collection=language_pg,
    )
    vision_submodule_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        language_hidden_size=hidden_size,
        pg_collection=vision_pg,
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": 50257},
        module_to_grid_map={"images": encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
    )

    torch.manual_seed(seed)
    model = MimoModel(mimo_config)
    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


# ============================================================================
# Checkpoint Test
# ============================================================================


def get_shared_tmpdir():
    """Create a shared temp directory across all ranks."""
    tmpdir_list = [None]
    if dist.get_rank() == 0:
        tmpdir_list[0] = tempfile.mkdtemp(prefix="mimo_ckpt_test_")
    dist.broadcast_object_list(tmpdir_list, src=0)
    return tmpdir_list[0]


def cleanup_tmpdir(tmpdir):
    """Clean up temp directory (rank 0 only)."""
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_checkpoint_test(
    encoder_tp,
    encoder_pp,
    encoder_dp,
    encoder_offset,
    llm_tp,
    llm_pp,
    llm_dp,
    llm_offset,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    validate_access_integrity=True,
):
    """Test distributed checkpoint save/load for non-colocated MiMo model."""
    rank = dist.get_rank()

    logger.info(f"[Rank {rank}] Creating grids...")
    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)

    # NOTE: We intentionally do NOT call parallel_state.initialize_model_parallel().
    # Non-colocated MiMo uses per-module process groups via HyperCommGrid,
    # not global parallel state. This ensures the test catches any code paths
    # that incorrectly fall back to parallel_state globals (Issue 2, NMFW-33).

    embedding_info = create_all_embedding_groups(encoder_grid, llm_grid)

    # --- Save checkpoint ---
    logger.info(f"[Rank {rank}] Creating model A (seed=1)...")
    model_a = create_mimo_model(
        encoder_grid,
        llm_grid,
        hidden_size,
        num_layers,
        vocab_size,
        seq_length,
        seed=1,
        embedding_info=embedding_info,
    )

    params_a = {name: param.clone() for name, param in model_a.named_parameters()}

    ckpt_dir = get_shared_tmpdir()
    try:
        logger.info(f"[Rank {rank}] Saving checkpoint to {ckpt_dir}...")

        sharded_sd_a = model_a.sharded_state_dict()
        save(sharded_sd_a, ckpt_dir, validate_access_integrity=validate_access_integrity)
        logger.info(f"[Rank {rank}] Checkpoint saved successfully")

        dist.barrier()

        # --- Load checkpoint into a fresh model ---
        logger.info(f"[Rank {rank}] Creating model B (seed=2)...")
        model_b = create_mimo_model(
            encoder_grid,
            llm_grid,
            hidden_size,
            num_layers,
            vocab_size,
            seq_length,
            seed=2,
            embedding_info=embedding_info,
        )

        # Verify model B has different weights before load
        for name, param in model_b.named_parameters():
            if name in params_a:
                assert not torch.equal(
                    param, params_a[name]
                ), f"Model B should have different weights before load: {name}"
                break

        logger.info(f"[Rank {rank}] Loading checkpoint...")
        sharded_sd_b = model_b.sharded_state_dict()
        state_dict, missing_keys, unexpected_keys = load(
            sharded_sd_b,
            ckpt_dir,
            validate_access_integrity=validate_access_integrity,
            strict=StrictHandling.RETURN_ALL,
        )

        # _extra_state mismatches are expected (TE FP8 metadata)
        real_missing = [k for k in missing_keys if '_extra_state' not in k]
        real_unexpected = [k for k in unexpected_keys if '_extra_state' not in k]
        assert len(real_missing) == 0, f"Missing keys (non-extra_state): {real_missing}"
        assert len(real_unexpected) == 0, f"Unexpected keys (non-extra_state): {real_unexpected}"

        model_b.load_state_dict(state_dict)
        logger.info(f"[Rank {rank}] Checkpoint loaded successfully")

        # --- Verify loaded weights match model A ---
        mismatches = []
        for name, param in model_b.named_parameters():
            if name in params_a and not torch.equal(param, params_a[name]):
                mismatches.append(name)

        assert len(mismatches) == 0, f"[Rank {rank}] Weight mismatch after load: {mismatches}"
        logger.info(f"[Rank {rank}] All {len(params_a)} parameters match after checkpoint load!")

    finally:
        cleanup_tmpdir(ckpt_dir)


def run_optimizer_checkpoint_test(
    encoder_tp,
    encoder_pp,
    encoder_dp,
    encoder_offset,
    llm_tp,
    llm_pp,
    llm_dp,
    llm_offset,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    validate_access_integrity=True,
):
    """Test distributed checkpoint save/load for MiMo model + optimizer state."""
    rank = dist.get_rank()

    logger.info(f"[Rank {rank}] Creating grids for optimizer test...")
    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)

    embedding_info = create_all_embedding_groups(encoder_grid, llm_grid)

    # --- Create model, wrap with DDP, and create optimizer ---
    model_a = create_mimo_model(
        encoder_grid,
        llm_grid,
        hidden_size,
        num_layers,
        vocab_size,
        seq_length,
        seed=1,
        embedding_info=embedding_info,
    )
    model_a.to(torch.device("cuda")).to(torch.bfloat16)

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, bucket_size=10000, use_distributed_optimizer=False
    )
    # Wrap submodules with DDP (required by get_megatron_optimizer)
    if model_a.language_model is not None:
        llm_pg = get_pg_collection(llm_grid)
        model_a.language_model = DistributedDataParallel(
            config=model_a.language_model.config,
            ddp_config=ddp_config,
            module=model_a.language_model,
            pg_collection=llm_pg,
        )
    for mod_name in list(model_a.modality_submodules.keys()):
        submod = model_a.modality_submodules[mod_name]
        if submod is not None:
            encoder_pg = get_pg_collection(encoder_grid)
            # Get config from the first encoder in the submodule
            first_encoder = next(iter(submod.encoders.values()))
            model_a.modality_submodules[mod_name] = DistributedDataParallel(
                config=first_encoder.config,
                ddp_config=ddp_config,
                module=submod,
                pg_collection=encoder_pg,
            )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optimizer_a = get_mimo_optimizer(model_a, opt_config)

    # Do a fake backward pass to populate optimizer state
    for param in model_a.parameters():
        param.grad = torch.randn_like(param)
    optimizer_a.step()

    # Snapshot optimizer state for comparison
    optim_state_a = {}
    for name, info in optimizer_a.module_infos.items():
        if info.is_active and info.optimizer:
            optim_state_a[name] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in info.optimizer.state_dict().items()
            }

    # Snapshot model params after optimizer step
    params_a = {name: param.clone() for name, param in model_a.named_parameters()}

    ckpt_dir = get_shared_tmpdir()
    try:
        logger.info(f"[Rank {rank}] Saving model + optimizer checkpoint...")

        # Save model state
        model_sd = model_a.sharded_state_dict()
        model_ckpt_dir = os.path.join(ckpt_dir, 'model')
        if rank == 0:
            os.makedirs(model_ckpt_dir, exist_ok=True)
        dist.barrier()
        save(model_sd, model_ckpt_dir, validate_access_integrity=validate_access_integrity)

        # Save optimizer state separately — per-module optimizer state has
        # different structure on different ranks, so it needs its own save
        # to avoid common state divergence (Issue 3, NMFW-33).
        # Need a fresh model sharded state dict (save() consumes tensor refs).
        model_sd_for_optim = model_a.sharded_state_dict()

        optim_sd = optimizer_a.sharded_state_dict(model_sd_for_optim, is_loading=False)
        optim_ckpt_dir = os.path.join(ckpt_dir, 'optimizer')
        if rank == 0:
            os.makedirs(optim_ckpt_dir, exist_ok=True)
        dist.barrier()
        save(optim_sd, optim_ckpt_dir, validate_access_integrity=False)

        logger.info(f"[Rank {rank}] Checkpoint saved successfully")

        dist.barrier()

        # --- Load into fresh model + optimizer ---
        logger.info(f"[Rank {rank}] Creating fresh model B + optimizer...")
        model_b = create_mimo_model(
            encoder_grid,
            llm_grid,
            hidden_size,
            num_layers,
            vocab_size,
            seq_length,
            seed=2,
            embedding_info=embedding_info,
        )
        model_b.to(torch.device("cuda")).to(torch.bfloat16)

        if model_b.language_model is not None:
            llm_pg_b = get_pg_collection(llm_grid)
            model_b.language_model = DistributedDataParallel(
                config=model_b.language_model.config,
                ddp_config=ddp_config,
                module=model_b.language_model,
                pg_collection=llm_pg_b,
            )
        for mod_name in list(model_b.modality_submodules.keys()):
            submod = model_b.modality_submodules[mod_name]
            if submod is not None:
                encoder_pg_b = get_pg_collection(encoder_grid)
                first_encoder = next(iter(submod.encoders.values()))
                model_b.modality_submodules[mod_name] = DistributedDataParallel(
                    config=first_encoder.config,
                    ddp_config=ddp_config,
                    module=submod,
                    pg_collection=encoder_pg_b,
                )

        optimizer_b = get_mimo_optimizer(model_b, opt_config)

        # Populate optimizer_b state so it has the right structure
        for param in model_b.parameters():
            param.grad = torch.randn_like(param)
        optimizer_b.step()

        logger.info(f"[Rank {rank}] Loading checkpoint...")

        # Build sharded state dict templates BEFORE loading, so tensor id()
        # mapping between model params and optimizer params is still valid.
        model_sd_b = model_b.sharded_state_dict()
        optim_sd_b = optimizer_b.sharded_state_dict(model_sd_b, is_loading=True)

        # Load model state
        loaded_model_sd, missing_keys, unexpected_keys = load(
            model_sd_b,
            model_ckpt_dir,
            validate_access_integrity=validate_access_integrity,
            strict=StrictHandling.RETURN_ALL,
        )
        real_missing = [k for k in missing_keys if '_extra_state' not in k]
        real_unexpected = [k for k in unexpected_keys if '_extra_state' not in k]
        assert len(real_missing) == 0, f"Missing keys: {real_missing}"
        assert len(real_unexpected) == 0, f"Unexpected keys: {real_unexpected}"
        model_b.load_state_dict(loaded_model_sd)

        # Load optimizer state
        loaded_optim_sd = load(optim_sd_b, optim_ckpt_dir, validate_access_integrity=False)

        optimizer_b.load_state_dict(loaded_optim_sd)

        logger.info(f"[Rank {rank}] Checkpoint loaded successfully")

        # --- Verify model params match ---
        model_mismatches = []
        for name, param in model_b.named_parameters():
            if name in params_a and not torch.equal(param, params_a[name]):
                model_mismatches.append(name)
        assert (
            len(model_mismatches) == 0
        ), f"[Rank {rank}] Model weight mismatch: {model_mismatches}"
        logger.info(f"[Rank {rank}] All {len(params_a)} model parameters match!")

        # --- Verify optimizer state matches ---
        keys_to_check = ['lr', 'weight_decay']
        for name, info in optimizer_b.module_infos.items():
            if info.is_active and info.optimizer and name in optim_state_a:
                loaded_optim = info.optimizer.state_dict()
                original_optim = optim_state_a[name]
                if 'optimizer' in loaded_optim and 'optimizer' in original_optim:
                    loaded_pg = loaded_optim['optimizer'].get('param_groups', [])
                    original_pg = original_optim['optimizer'].get('param_groups', [])
                    assert len(loaded_pg) == len(original_pg), (
                        f"[Rank {rank}] Optimizer {name}: param_groups count mismatch "
                        f"{len(loaded_pg)} != {len(original_pg)}"
                    )
                    for i, (lpg, opg) in enumerate(zip(loaded_pg, original_pg)):
                        for key in keys_to_check:
                            if key in opg and key in lpg:
                                assert lpg[key] == opg[key], (
                                    f"[Rank {rank}] Optimizer {name} param_group[{i}].{key} "
                                    f"mismatch: {lpg[key]} != {opg[key]}"
                                )
        logger.info(f"[Rank {rank}] Optimizer state verified!")

    finally:
        cleanup_tmpdir(ckpt_dir)


# ============================================================================
# Test Configurations
# ============================================================================


TEST_CONFIGS = {
    2: [
        {
            "name": "baseline_2gpu",
            "encoder_tp": 1,
            "encoder_pp": 1,
            "encoder_dp": 1,
            "encoder_offset": 0,
            "llm_tp": 1,
            "llm_pp": 1,
            "llm_dp": 1,
            "llm_offset": 1,
            "hidden_size": 256,
            "num_layers": 2,
            "vocab_size": 1000,
            "seq_length": 64,
        }
    ],
    4: [
        {
            "name": "llm_pp3_4gpu",
            "encoder_tp": 1,
            "encoder_pp": 1,
            "encoder_dp": 1,
            "encoder_offset": 0,
            "llm_tp": 1,
            "llm_pp": 3,
            "llm_dp": 1,
            "llm_offset": 1,
            "hidden_size": 256,
            "num_layers": 3,
            "vocab_size": 1000,
            "seq_length": 64,
        }
    ],
    8: [
        {
            "name": "encoder_tp2_llm_tp2_pp3_8gpu",
            "encoder_tp": 2,
            "encoder_pp": 1,
            "encoder_dp": 1,
            "encoder_offset": 0,
            "llm_tp": 2,
            "llm_pp": 3,
            "llm_dp": 1,
            "llm_offset": 2,
            "hidden_size": 256,
            "num_layers": 3,
            "vocab_size": 1000,
            "seq_length": 64,
        },
        {
            "name": "encoder_tp1_llm_tp1_pp7_8gpu",
            "encoder_tp": 1,
            "encoder_pp": 1,
            "encoder_dp": 1,
            "encoder_offset": 0,
            "llm_tp": 1,
            "llm_pp": 7,
            "llm_dp": 1,
            "llm_offset": 1,
            "hidden_size": 256,
            "num_layers": 7,
            "vocab_size": 1000,
            "seq_length": 64,
        },
        {
            "name": "encoder_tp2_pp2_llm_tp2_pp2_8gpu",
            "encoder_tp": 2,
            "encoder_pp": 2,
            "encoder_dp": 1,
            "encoder_offset": 0,
            "llm_tp": 2,
            "llm_pp": 2,
            "llm_dp": 1,
            "llm_offset": 4,
            "hidden_size": 256,
            "num_layers": 2,
            "vocab_size": 1000,
            "seq_length": 64,
        },
    ],
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    logger.info(f"Rank {rank}/{world_size} initialized")

    configs = TEST_CONFIGS.get(world_size, [])
    if not configs:
        logger.error(
            f"No configs for world_size={world_size}. Available: {list(TEST_CONFIGS.keys())}"
        )
        dist.destroy_process_group()
        return

    if args.config:
        configs = [c for c in configs if c["name"] == args.config]

    for config in configs:
        name = config.pop("name")

        # Model-only checkpoint test
        logger.info(f"Running model checkpoint test: {name}")
        try:
            run_checkpoint_test(**config, validate_access_integrity=not args.no_validate)
            logger.info(f"Model checkpoint test {name} PASSED")
        except Exception as e:
            logger.error(f"Model checkpoint test {name} FAILED: {e}")
            raise
        finally:
            config["name"] = name

        dist.barrier()

        # Model + optimizer checkpoint test
        name = config.pop("name")
        logger.info(f"Running optimizer checkpoint test: {name}")
        try:
            run_optimizer_checkpoint_test(**config, validate_access_integrity=not args.no_validate)
            logger.info(f"Optimizer checkpoint test {name} PASSED")
        except Exception as e:
            logger.error(f"Optimizer checkpoint test {name} FAILED: {e}")
            raise
        finally:
            config["name"] = name

    dist.destroy_process_group()
    logger.info("All checkpoint tests completed successfully!")


if __name__ == "__main__":
    main()
