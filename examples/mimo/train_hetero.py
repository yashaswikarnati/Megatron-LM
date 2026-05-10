# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone heterogeneous MIMO mock training loop.

This entrypoint is intentionally separate from examples/mimo/train.py. The
standard Megatron pretrain path owns a single homogeneous model-parallel
topology, while this loop owns one HyperCommGrid per MIMO module and wires the
multi-module pipeline schedule directly.
"""

import argparse
import os
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TELayerNormColumnParallelLinear = None
    TERowParallelLinear = None


MOCK_MODEL_PRESET = "mock"
NEMOTRON_20L_MODEL_PRESET = "nemotron-moe-vlm-20l"
NEMOTRON_20L_HYBRID_PATTERN = "MEMEM*EMEMEM*EMEMEM*"
NEMOTRON_20L_IMAGE_SEQ_PER_TILE = 256
NEMOTRON_20L_MAX_NUM_TILES = 12
NEMOTRON_20L_DEFAULT_STAGE = "stage2"

_ACTIVE_GRIDS: list[HyperCommGrid] = []
_EMBEDDING_PG_CACHE: dict[tuple[int, ...], tuple[dist.ProcessGroup, dist.ProcessGroup]] = {}


def parse_args() -> argparse.Namespace:
    """Parse standalone hetero MIMO loop arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    grid = parser.add_argument_group("module grids")
    grid.add_argument("--encoder-offset", type=int, default=0)
    grid.add_argument("--encoder-tp", type=int, default=2)
    grid.add_argument("--encoder-cp", type=int, default=1)
    grid.add_argument("--encoder-pp", type=int, default=2)
    grid.add_argument("--encoder-dp", type=int, default=1)
    grid.add_argument("--encoder-ep", type=int, default=1)
    grid.add_argument("--encoder-expt-tp", type=int, default=None)
    grid.add_argument("--encoder-expt-dp", type=int, default=None)
    grid.add_argument("--llm-offset", type=int, default=4)
    grid.add_argument("--llm-tp", type=int, default=1)
    grid.add_argument("--llm-cp", type=int, default=1)
    grid.add_argument("--llm-pp", type=int, default=2)
    grid.add_argument("--llm-dp", type=int, default=2)
    grid.add_argument("--llm-ep", type=int, default=2)
    grid.add_argument("--llm-expt-tp", type=int, default=1)
    grid.add_argument("--llm-expt-dp", type=int, default=1)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-preset",
        choices=[MOCK_MODEL_PRESET, NEMOTRON_20L_MODEL_PRESET],
        default=MOCK_MODEL_PRESET,
        help="Model config preset. The Nemotron preset matches the 20L reference script.",
    )
    model.add_argument("--hidden-size", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--vocab-size", type=int, default=512)
    model.add_argument("--seq-length", type=int, default=32)
    model.add_argument("--image-seq-length", type=int, default=None)
    model.add_argument("--image-token-id", type=int, default=511)
    model.add_argument("--pad-token-id", type=int, default=0)
    model.add_argument("--image-token", type=str, default="<image>")
    model.add_argument("--tokenizer-model", type=str, default=None)
    model.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    model.add_argument("--image-tag-type", type=str, default="")
    model.add_argument("--force-system-message", action="store_true")
    model.add_argument("--num-moe-experts", type=int, default=4)
    model.add_argument("--moe-router-topk", type=int, default=1)
    model.add_argument("--moe-grouped-gemm", action="store_true")
    model.add_argument("--img-h", type=int, default=512)
    model.add_argument("--img-w", type=int, default=512)
    model.add_argument("--patch-dim", type=int, default=16)
    model.add_argument("--class-token-len", type=int, default=8)
    model.add_argument("--num-image-tiles", type=int, default=NEMOTRON_20L_MAX_NUM_TILES)
    model.add_argument("--freeze-lm", action="store_true", help="Freeze language model params")
    model.add_argument("--freeze-vit", action="store_true", help="Freeze vision encoder params")
    model.add_argument(
        "--freeze-projection", action="store_true", help="Freeze vision projection params"
    )
    model.add_argument(
        "--training-stage",
        choices=["stage1", "stage2", "stage3"],
        default=None,
        help="Nemotron VLM freeze stage. Defaults to stage2 for the 20L preset.",
    )
    model.add_argument(
        "--fp32", action="store_true", help="Build and train in fp32 instead of bf16"
    )

    train = parser.add_argument_group("training")
    train.add_argument("--micro-batch-size", type=int, default=2)
    train.add_argument("--global-batch-size", type=int, default=None)
    train.add_argument("--num-microbatches", type=int, default=2)
    train.add_argument("--train-iters", type=int, default=2)
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--min-lr", type=float, default=None)
    train.add_argument("--lr-decay-style", type=str, default="constant")
    train.add_argument("--lr-warmup-iters", type=int, default=0)
    train.add_argument("--lr-decay-iters", type=int, default=None)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DDP gradient-reduce overlap. Disable for parity with the 20L reference script.",
    )
    train.add_argument(
        "--ddp-bucket-size",
        type=int,
        default=10000,
        help="DDP bucket size. Use 0 for a single unbounded bucket.",
    )
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--log-interval", type=int, default=1)

    return parser.parse_args()


def clear_transformer_engine_env() -> None:
    """Clear attention backend overrides that can conflict with GPTModel construction."""
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)


def initialize_distributed() -> None:
    """Initialize torch.distributed for torchrun."""
    clear_transformer_engine_env()
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    try:
        parallel_state.get_global_memory_buffer()
    except AssertionError:
        parallel_state._set_global_memory_buffer()
    dist.barrier()


def print_rank_0(message: str) -> None:
    """Print only on global rank zero."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()


def debug_rank(message: str) -> None:
    """Emit per-rank startup checkpoints when MIMO_HETERO_DEBUG is set."""
    if os.environ.get("MIMO_HETERO_DEBUG"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        sys.stderr.write(f"[rank {rank}] {message}\n")
        sys.stderr.flush()


def is_nemotron_20l(args: argparse.Namespace) -> bool:
    """Return whether the run should use the Nemotron6-MoE VLM 20L architecture."""
    return args.model_preset == NEMOTRON_20L_MODEL_PRESET


def apply_model_preset(args: argparse.Namespace) -> None:
    """Apply architecture defaults for the selected model preset."""
    if not is_nemotron_20l(args):
        return

    args.num_layers = 20
    args.hidden_size = 2688
    args.num_attention_heads = 32
    args.num_moe_experts = 128
    args.moe_router_topk = 6
    args.moe_grouped_gemm = True
    args.seq_length = 8192
    args.image_seq_length = NEMOTRON_20L_IMAGE_SEQ_PER_TILE * args.num_image_tiles


def apply_training_stage(args: argparse.Namespace) -> None:
    """Apply the reference Nemotron VLM freeze stage defaults."""
    if not is_nemotron_20l(args):
        return

    stage = args.training_stage or NEMOTRON_20L_DEFAULT_STAGE
    if stage == "stage1":
        args.freeze_vit = True
        args.freeze_lm = True
    elif stage == "stage2":
        args.freeze_vit = True
    elif stage != "stage3":
        raise ValueError(f"unsupported Nemotron VLM training stage: {stage}")
    args.training_stage = stage


def resolve_image_token_id(args: argparse.Namespace) -> None:
    """Resolve the image token id from the reference MultimodalTokenizer when provided."""
    if not is_nemotron_20l(args) or not args.tokenizer_model:
        return

    from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
        MegatronMultimodalTokenizer,
    )

    tokenizer = MegatronMultimodalTokenizer(
        path=args.tokenizer_model,
        prompt_format=args.tokenizer_prompt_format,
        special_tokens=[args.image_token],
        image_tag_type=args.image_tag_type,
        force_system_message=args.force_system_message,
    )
    image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    if image_token_id is None:
        raise RuntimeError(
            f"tokenizer at {args.tokenizer_model} did not produce an id for {args.image_token}"
        )
    args.image_token_id = int(image_token_id)
    if tokenizer.pad is not None:
        args.pad_token_id = int(tokenizer.pad)
    if tokenizer.vocab_size is not None:
        args.vocab_size = int(tokenizer.vocab_size)


def validate_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Validate the Phase 2 non-colocated 1F1B mock-training layout."""
    if args.encoder_cp != 1 or args.llm_cp != 1:
        raise ValueError("Phase 2 mock training currently supports CP=1 only")
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads")
    if args.num_moe_experts > 0 and args.num_moe_experts % args.llm_ep != 0:
        raise ValueError("--num-moe-experts must be divisible by --llm-ep")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1")
    if not 0 <= args.image_token_id < args.vocab_size:
        raise ValueError("--image-token-id must be within --vocab-size")
    if not 0 <= args.pad_token_id < args.vocab_size:
        raise ValueError("--pad-token-id must be within --vocab-size")

    image_seq_length = args.image_seq_length or args.seq_length // 2
    if image_seq_length >= args.seq_length:
        raise ValueError("--image-seq-length must be smaller than --seq-length")
    if args.seq_length - image_seq_length < 2:
        raise ValueError("mock next-token training needs at least two text tokens")
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("--micro-batch-size * --llm-dp must be divisible by --encoder-dp")

    encoder_size = args.encoder_tp * args.encoder_cp * args.encoder_pp * args.encoder_dp
    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    encoder_ranks = set(range(args.encoder_offset, args.encoder_offset + encoder_size))
    llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
    all_ranks = set(range(world_size))

    if not encoder_ranks.isdisjoint(llm_ranks):
        raise ValueError(
            "Phase 2 train_hetero.py supports non-colocated 1F1B only; "
            f"module rank spans overlap at {sorted(encoder_ranks & llm_ranks)}"
        )
    if encoder_ranks | llm_ranks != all_ranks:
        raise ValueError(
            "The non-colocated module grids must cover every torchrun rank exactly once; "
            f"covered={sorted(encoder_ranks | llm_ranks)}, world={sorted(all_ranks)}"
        )

    return encoder_size, llm_size


def is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Return whether pg is a real process group for this rank."""
    group_member = getattr(dist, "GroupMember", None)
    non_member = getattr(group_member, "NON_GROUP_MEMBER", None)
    return pg is not None and pg != non_member


def destroy_process_group_if_member(pg: Optional[dist.ProcessGroup]) -> None:
    """Destroy pg when this rank owns a process-group handle."""
    if is_process_group_member(pg):
        dist.destroy_process_group(pg)


def create_hypercomm_grid(
    offset: int,
    tp: int,
    cp: int,
    pp: int,
    dp: int,
    ep: int,
    expt_tp: Optional[int],
    expt_dp: Optional[int],
) -> HyperCommGrid:
    """Create a dense grid plus expert layout and required process groups."""
    expt_tp = tp if expt_tp is None else expt_tp
    module_world_size = tp * cp * pp * dp
    expert_model_size = expt_tp * ep * pp
    if module_world_size % expert_model_size != 0:
        raise ValueError(
            f"module_world_size ({module_world_size}) must be divisible by "
            f"expt_tp*ep*pp ({expert_model_size})"
        )
    if expt_dp is None:
        expt_dp = module_world_size // expert_model_size
    if expt_tp * ep * expt_dp * pp != module_world_size:
        raise ValueError(
            f"expt_tp*ep*expt_dp*pp ({expt_tp * ep * expt_dp * pp}) must equal "
            f"module_world_size ({module_world_size})"
        )

    grid = HyperCommGrid(
        shape=[tp, cp, dp, pp],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.register_layout(
        "expert",
        [expt_tp, ep, expt_dp, pp],
        ["expt_tp", "ep", "expt_dp", "pp"],
        aliases={"tp_ep": ["expt_tp", "ep"], "tp_ep_pp": ["expt_tp", "ep", "pp"]},
    )

    for dims in (
        ["tp"],
        ["cp"],
        ["pp"],
        ["dp"],
        ["dp", "cp"],
        ["tp", "cp"],
        ["ep"],
        ["expt_tp"],
        ["expt_dp"],
        ["tp", "pp"],
        ["tp", "cp", "dp"],
        ["tp", "cp", "pp", "dp"],
        "tp_ep",
        "tp_ep_pp",
    ):
        grid.create_pg(dims)

    _ACTIVE_GRIDS.append(grid)
    return grid


def destroy_runtime_process_groups() -> None:
    """Destroy process groups created by this script."""
    destroyed_embedding_pgs = set()
    for pos_embd_pg, embd_pg in _EMBEDDING_PG_CACHE.values():
        for pg in (pos_embd_pg, embd_pg):
            if id(pg) in destroyed_embedding_pgs:
                continue
            destroy_process_group_if_member(pg)
            destroyed_embedding_pgs.add(id(pg))
    _EMBEDDING_PG_CACHE.clear()

    for grid in _ACTIVE_GRIDS:
        grid.destroy()
    _ACTIVE_GRIDS.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def get_pg_collection(grid: HyperCommGrid) -> ProcessGroupCollection:
    """Build a ProcessGroupCollection from a populated HyperCommGrid."""
    return ProcessGroupCollection.from_hyper_comm_grid(
        grid,
        required_pgs=[
            "tp",
            "cp",
            "pp",
            "dp",
            "dp_cp",
            "tp_cp",
            "mp",
            "tp_dp_cp",
            "ep",
            "expt_tp",
            "expt_dp",
            "tp_ep",
            "tp_ep_pp",
            "intra_dist_opt",
        ],
    )


def create_all_embedding_groups(grids: list[HyperCommGrid]) -> None:
    """Create PP-derived embedding groups in a consistent global order."""
    pp_rank_sets: list[tuple[int, ...]] = []
    seen_pp_rank_sets = set()
    for grid in sorted(grids, key=lambda candidate: (candidate.rank_offset, candidate.size)):
        for pp_ranks in grid.get_rank_enum("pp"):
            pp_rank_tuple = tuple(pp_ranks)
            if pp_rank_tuple in seen_pp_rank_sets:
                continue
            pp_rank_sets.append(pp_rank_tuple)
            seen_pp_rank_sets.add(pp_rank_tuple)

    for pp_ranks in pp_rank_sets:
        if pp_ranks not in _EMBEDDING_PG_CACHE:
            pos_embd_ranks = [pp_ranks[0]]
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            _EMBEDDING_PG_CACHE[pp_ranks] = (
                dist.new_group(ranks=pos_embd_ranks),
                dist.new_group(ranks=embd_ranks),
            )


def add_embedding_groups(
    pg_collection: ProcessGroupCollection, is_language_model: bool = False
) -> ProcessGroupCollection:
    """Attach cached embedding process groups to a ProcessGroupCollection."""
    if not is_process_group_member(getattr(pg_collection, "pp", None)):
        return pg_collection

    pp_ranks = tuple(dist.get_process_group_ranks(pg_collection.pp))
    pos_embd_pg, embd_pg = _EMBEDDING_PG_CACHE[pp_ranks]

    pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
    if is_language_model:
        pg_collection.embd = (
            embd_pg
            if is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp)
            else None
        )
    else:
        pg_collection.embd = None

    return pg_collection


def get_pg_collection_with_embedding_groups(
    grid: HyperCommGrid, is_language_model: bool = False
) -> ProcessGroupCollection:
    """Build a ProcessGroupCollection and add PP-derived embedding groups."""
    return add_embedding_groups(get_pg_collection(grid), is_language_model=is_language_model)


def is_rank_in_grid(grid: HyperCommGrid) -> bool:
    """Return whether this global rank is inside a grid's rank span."""
    rank = dist.get_rank()
    return grid.rank_offset <= rank < grid.rank_offset + grid.size


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return a base-layout dimension size."""
    return grid.shape[grid.dim_names.index(dim)]


def get_group_size_or(pg: Optional[dist.ProcessGroup], fallback: int) -> int:
    """Return pg size on member ranks, otherwise fallback."""
    return pg.size() if is_process_group_member(pg) else fallback


def get_group_rank_or(pg: Optional[dist.ProcessGroup], fallback: int = 0) -> int:
    """Return rank inside pg on member ranks, otherwise fallback."""
    return dist.get_rank(pg) if is_process_group_member(pg) else fallback


def set_module_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool) -> None:
    """Set requires_grad for every parameter in a module when the module exists."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_model_init_seed(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_offset: int
):
    """Seed CPU model init consistently across TP/DP peers for one module role."""
    pp_rank = get_group_rank_or(getattr(pg_collection, "pp", None))
    torch.manual_seed(args.seed + role_offset + (100 * pp_rank))


def initialize_model_parallel_rng(args: argparse.Namespace, pg_collection: ProcessGroupCollection):
    """Initialize CUDA RNG tracker using the active module's hetero process groups."""
    pp_rank = get_group_rank_or(getattr(pg_collection, "pp", None))
    tp_rank = get_group_rank_or(getattr(pg_collection, "tp", None))
    ep_rank = get_group_rank_or(getattr(pg_collection, "ep", None))
    expt_tp_rank = get_group_rank_or(getattr(pg_collection, "expt_tp", None))
    model_parallel_cuda_manual_seed(
        args.seed + (100 * pp_rank),
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        etp_rank=expt_tp_rank,
        force_reset_rng=True,
    )


def active_ddp_modules(mimo_model: MimoModel) -> list[DistributedDataParallel]:
    """Return active DDP-wrapped submodules owned by this rank."""
    modules = []
    if isinstance(mimo_model.language_model, DistributedDataParallel):
        modules.append(mimo_model.language_model)
    modules.extend(
        submodule
        for submodule in mimo_model.modality_submodules.values()
        if isinstance(submodule, DistributedDataParallel)
    )
    return modules


def broadcast_active_params(mimo_model: MimoModel) -> None:
    """Synchronize initial parameters across each module's DP groups."""
    for module in active_ddp_modules(mimo_model):
        module.broadcast_params()


def zero_active_grad_buffers(mimo_model: MimoModel) -> None:
    """Clear MCore DDP grad buffers before each training iteration."""
    for module in active_ddp_modules(mimo_model):
        module.zero_grad_buffer()


def get_grid_coordinate(grid: HyperCommGrid, dim: str) -> int:
    """Return this rank's coordinate for a base-layout dimension."""
    if not is_rank_in_grid(grid):
        return 0

    local_rank = dist.get_rank() - grid.rank_offset
    coordinates = {}
    for dim_name, dim_size in zip(grid.dim_names, grid.shape):
        coordinates[dim_name] = local_rank % dim_size
        local_rank //= dim_size
    return coordinates[dim]


def get_mock_data_seed(
    args: argparse.Namespace, grid: HyperCommGrid, module_seed_offset: int
) -> int:
    """Seed mock data by data-parallel lane so PP/TP stages see coherent batches."""
    dp_lane = get_grid_coordinate(grid, "dp") if "dp" in grid.dim_names else 0
    return args.seed + module_seed_offset + dp_lane


def build_no_sync_func(mimo_model: MimoModel):
    """Build a no_sync context spanning all active MIMO submodules."""

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    return no_sync_func


class RADIOEncoderWrapper(torch.nn.Module):
    """RADIO encoder wrapper matching the Nemotron6-MoE VLM provider."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection],
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
    ) -> None:
        super().__init__()
        self.class_token_len = class_token_len
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle
        self.force_eval_mode = force_eval_mode
        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            pg_collection=pg_collection,
        )
        if self.force_eval_mode:
            self.radio_model.eval()

    def train(self, mode: bool = True):
        """Keep frozen RADIO in eval mode while allowing the projection to train."""
        super().train(mode)
        if self.force_eval_mode:
            self.radio_model.eval()
        return self

    @property
    def config(self):
        """Expose the underlying RADIO config for DDP wrapping."""
        return self.radio_model.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run RADIO, drop class tokens, and apply pixel shuffle."""
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        debug_rank(f"RADIO forward start: input_shape={tuple(x.shape)}")
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            embeddings = self.radio_model(x)
        debug_rank(f"RADIO forward done: output_shape={tuple(embeddings.shape)}")
        if self.drop_class_token:
            embeddings = embeddings[:, self.class_token_len :, :]
            debug_rank(f"RADIO class tokens dropped: output_shape={tuple(embeddings.shape)}")
        if self.apply_pixel_shuffle:
            embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
            debug_rank(f"RADIO pixel shuffle done: output_shape={tuple(embeddings.shape)}")
        return embeddings

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate checkpoint sharding to the wrapped RADIO model."""
        sharded_sd = {}
        for name, child in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(child, f"{prefix}{name}.", sharded_offsets, metadata)
            )
        return sharded_sd


def projection_layer_spec() -> ModuleSpec:
    """Return the TE-backed projection MLP spec."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def nemotron_projection_layer_spec() -> ModuleSpec:
    """Return the Nemotron VLM RADIO-to-language projector layer spec."""
    if TELayerNormColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TELayerNormColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def nemotron_language_config(
    args: argparse.Namespace, tp_size: int, pp_size: int, ep_size: int, expt_tp_size: int
) -> TransformerConfig:
    """Build the exact Nemotron6-MoE 20L language TransformerConfig."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=20,
        hidden_size=2688,
        num_attention_heads=32,
        attention_backend=AttnBackend.flash,
        num_query_groups=2,
        ffn_hidden_size=1856,
        kv_channels=128,
        activation_func=squared_relu,
        gated_linear_unit=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        init_method_std=0.0173,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=expt_tp_size,
        sequence_parallel=tp_size > 1,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
        calculate_per_token_loss=True,
        bias_activation_fusion=False,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=False,
        recompute_granularity="selective",
        recompute_modules=["core_attn"],
        moe_ffn_hidden_size=1856,
        num_moe_experts=128,
        moe_router_topk=6,
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.0001,
        moe_shared_expert_intermediate_size=3712,
        moe_shared_expert_overlap=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        use_fused_weighted_squared_relu=True,
        is_hybrid_model=True,
        mamba_num_heads=64,
        mamba_head_dim=64,
    )
    config.position_embedding_type = "none"
    config.seq_length = 8192
    config.max_position_embeddings = 8192
    return config


def require_per_token_loss(config: TransformerConfig) -> None:
    """The hetero MIMO loop scales both language and vision grads by real LM tokens."""
    if not config.calculate_per_token_loss:
        raise ValueError("train_hetero.py requires calculate_per_token_loss=True")


def radio_vision_config(args: argparse.Namespace, tp_size: int, pp_size: int) -> TransformerConfig:
    """Build the exact RADIO vision TransformerConfig from the 20L reference provider."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
    config.kv_channels = 80
    config.num_query_groups = 16
    config.ffn_hidden_size = 5120
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1.0e-6
    config.layernorm_zero_centered_gamma = False
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    return config


def nemotron_projection_config(args: argparse.Namespace, tp_size: int) -> TransformerConfig:
    """Build the exact RADIO-to-Nemotron projection config."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2688,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
    config.tensor_model_parallel_size = tp_size
    config.ffn_hidden_size = 4 * 5120
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    config.activation_func = squared_relu
    config.normalization = "RMSNorm"
    return config


def language_model_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    llm_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the language ModuleSpec for the local language grid."""
    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    ep_pg = getattr(pg_collection, "ep", None) if pg_collection is not None else None
    expt_tp_pg = getattr(pg_collection, "expt_tp", None) if pg_collection is not None else None

    fallback_tp_size = get_grid_dim_size(llm_grid, "tp")
    pp_rank = get_group_rank_or(pp_pg)
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(llm_grid, "pp"))
    tp_size = get_group_size_or(tp_pg, fallback_tp_size)
    ep_size = get_group_size_or(ep_pg, args.llm_ep)
    expt_tp_size = get_group_size_or(expt_tp_pg, args.llm_expt_tp or fallback_tp_size)
    if is_nemotron_20l(args):
        config = nemotron_language_config(args, tp_size, pp_size, ep_size, expt_tp_size)
        require_per_token_loss(config)
        return ModuleSpec(
            module=MambaModel,
            params={
                "config": config,
                "mamba_stack_spec": mamba_stack_spec,
                "vocab_size": args.vocab_size,
                "max_sequence_length": args.seq_length,
                "pre_process": pp_rank == 0,
                "post_process": pp_rank == pp_size - 1,
                "hybrid_override_pattern": NEMOTRON_20L_HYBRID_PATTERN,
                "position_embedding_type": "none",
                "scatter_embedding_sequence_parallel": False,
                "pg_collection": pg_collection,
            },
        )

    num_moe_experts = args.num_moe_experts if args.num_moe_experts > 0 else None
    bf16 = not args.fp32

    moe_kwargs = {}
    if num_moe_experts is not None:
        moe_kwargs = {
            "num_moe_experts": num_moe_experts,
            "moe_router_topk": args.moe_router_topk,
            "moe_router_pre_softmax": args.moe_router_topk == 1,
            "expert_model_parallel_size": ep_size,
            "expert_tensor_parallel_size": expt_tp_size,
            "moe_grouped_gemm": args.moe_grouped_gemm,
        }

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        **moe_kwargs,
    )
    require_per_token_loss(config)
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_moe_experts, moe_grouped_gemm=args.moe_grouped_gemm
            ),
            "vocab_size": args.vocab_size,
            "max_sequence_length": args.seq_length,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
            "pg_collection": pg_collection,
        },
    )


def vision_submodules_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    encoder_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the vision ModuleSpec for the local encoder grid."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    tp_size = get_group_size_or(tp_pg, get_grid_dim_size(encoder_grid, "tp"))
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(encoder_grid, "pp"))
    pp_rank = get_group_rank_or(pp_pg)
    bf16 = not args.fp32

    if is_nemotron_20l(args):
        vision_config = radio_vision_config(args, tp_size, pp_size)
        vision_encoder_spec = ModuleSpec(
            module=RADIOEncoderWrapper,
            params={
                "transformer_config": vision_config,
                "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
                "pg_collection": pg_collection,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "patch_dim": args.patch_dim,
                "class_token_len": args.class_token_len,
                "drop_class_token": True,
                "apply_pixel_shuffle": True,
                "force_eval_mode": args.freeze_vit,
            },
        )
        vision_projection_spec = ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": nemotron_projection_config(args, tp_size),
                "submodules": nemotron_projection_layer_spec().submodules,
                "projector_type": "mlp",
                "input_size": 5120,
                "tp_group": tp_pg if is_process_group_member(tp_pg) else None,
            },
        )
        return ModuleSpec(
            module=VisionModalitySubmodules,
            params={"pg_collection": pg_collection},
            submodules={
                "encoders": {"radio_encoder": vision_encoder_spec},
                "input_projections": [vision_projection_spec],
            },
        )

    vision_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        calculate_per_token_loss=True,
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

    projection_config = TransformerConfig(
        num_layers=1, hidden_size=args.hidden_size, num_attention_heads=1
    )
    projection_config.ffn_hidden_size = args.hidden_size
    projection_config.activation_func = torch.nn.functional.gelu

    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "tp_group": tp_pg if is_process_group_member(tp_pg) else None,
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


def build_mimo_model(
    args: argparse.Namespace,
    encoder_grid: HyperCommGrid,
    llm_grid: HyperCommGrid,
    encoder_name: str,
):
    """Build the MIMO model and wrap active modules in MCore DDP."""
    language_pg = get_pg_collection_with_embedding_groups(llm_grid, is_language_model=True)
    vision_pg = get_pg_collection_with_embedding_groups(encoder_grid, is_language_model=False)
    rank_in_language_grid = is_rank_in_grid(llm_grid)
    rank_in_encoder_grid = is_rank_in_grid(encoder_grid)
    debug_rank(
        "building model specs "
        f"rank_in_encoder={rank_in_encoder_grid} rank_in_language={rank_in_language_grid}"
    )
    if rank_in_language_grid:
        set_model_init_seed(args, language_pg, role_offset=20_000)
        initialize_model_parallel_rng(args, language_pg)
    elif rank_in_encoder_grid:
        set_model_init_seed(args, vision_pg, role_offset=10_000)
        initialize_model_parallel_rng(args, vision_pg)

    module_to_grid_map = {encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid}
    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec(
            args, language_pg if rank_in_language_grid else None, llm_grid
        ),
        modality_submodules_spec={
            encoder_name: vision_submodules_spec(
                args, vision_pg if rank_in_encoder_grid else None, encoder_grid
            )
        },
        special_token_ids={encoder_name: args.image_token_id},
        module_to_grid_map=module_to_grid_map,
    )

    debug_rank("constructing MimoModel")
    mimo_model = MimoModel(
        mimo_config,
        cp_group=language_pg.cp if rank_in_language_grid else None,
        tp_group=language_pg.tp if rank_in_language_grid else None,
    )
    debug_rank("moving MimoModel to cuda")
    mimo_model.to(torch.device("cuda"))
    if not args.fp32:
        mimo_model.to(torch.bfloat16)
    debug_rank("MimoModel moved to target dtype/device")

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=args.overlap_grad_reduce,
        bucket_size=args.ddp_bucket_size if args.ddp_bucket_size > 0 else None,
        use_distributed_optimizer=True,
    )
    if mimo_model.language_model is not None:
        if args.freeze_lm:
            set_module_requires_grad(mimo_model.language_model, False)
        debug_rank("wrapping language model in DDP")
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=language_pg,
        )
        debug_rank("language model DDP ready")

    if encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[encoder_name]
        if submodule is not None:
            encoder_module_name = "radio_encoder" if is_nemotron_20l(args) else "clip_encoder"
            if args.freeze_vit:
                set_module_requires_grad(submodule.encoders[encoder_module_name], False)
            if args.freeze_projection:
                for projection in submodule.input_projections:
                    set_module_requires_grad(projection, False)
            debug_rank("wrapping vision submodule in DDP")
            mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                config=submodule.encoders[encoder_module_name].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=vision_pg,
            )
            debug_rank("vision submodule DDP ready")

    broadcast_active_params(mimo_model)
    return mimo_model, module_to_grid_map, language_pg, vision_pg


class MockVLMIterator:
    """Infinite iterator yielding synthetic VLM-like microbatches."""

    def __init__(
        self, args: argparse.Namespace, micro_batch_size: int, encoder_name: str, seed: int
    ) -> None:
        self.args = args
        self.micro_batch_size = micro_batch_size
        self.encoder_name = encoder_name
        self.image_seq_length = args.image_seq_length or args.seq_length // 2
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

        if is_nemotron_20l(args):
            encoder_inputs = {
                "radio_encoder": {
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
                "clip_encoder": {"hidden_states": encoder_hidden_states, "attention_mask": None}
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


def wire_training_hooks(
    mimo_model: MimoModel,
    language_pg: ProcessGroupCollection,
    vision_pg: ProcessGroupCollection,
    token_count_group: dist.ProcessGroup,
) -> None:
    """Attach MIMO-specific grad sync hooks expected by the pipeline schedule."""

    def is_token_source_rank() -> bool:
        return (
            is_process_group_member(getattr(language_pg, "pp", None))
            and is_process_group_member(getattr(language_pg, "tp", None))
            and is_pp_last_stage(language_pg.pp)
            and language_pg.tp.rank() == 0
        )

    def finalize_grads_func(_model_list, num_tokens, force_all_reduce=False, **_kwargs):
        if num_tokens is None:
            raise RuntimeError("train_hetero.py expects calculate_per_token_loss=True")

        token_count = torch.zeros(1, dtype=torch.float32, device="cuda")
        if is_token_source_rank():
            token_count[0] = num_tokens.to(device="cuda", dtype=torch.float32).sum()
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=token_count_group)
        global_num_tokens = token_count.item()

        if mimo_model.language_model is not None:
            debug_rank("finalizing language grads")
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
            debug_rank("language grads finalized")
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                debug_rank("finalizing vision grads")
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=vision_pg,
                    force_all_reduce=force_all_reduce,
                )
                debug_rank("vision grads finalized")

        if global_num_tokens > 0:
            scale = 1.0 / global_num_tokens
            if mimo_model.language_model is not None:
                debug_rank("scaling language grads")
                mimo_model.language_model.scale_gradients(scale)
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    debug_rank("scaling vision grads")
                    submodule.scale_gradients(scale)

    mimo_model.config.no_sync_func = build_no_sync_func(mimo_model)
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device="cuda", requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )


def select_data_iterator(
    args: argparse.Namespace,
    encoder_grid: HyperCommGrid,
    llm_grid: HyperCommGrid,
    encoder_name: str,
) -> Optional[MockVLMIterator]:
    """Create the per-role data iterator needed by local ranks."""
    llm_mbs = args.micro_batch_size
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")
    encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp

    encoder_needs_data = is_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )

    if encoder_needs_data and not llm_needs_data:
        return MockVLMIterator(
            args,
            encoder_mbs,
            encoder_name,
            get_mock_data_seed(args, encoder_grid, module_seed_offset=0),
        )
    if llm_needs_data and not encoder_needs_data:
        return MockVLMIterator(
            args,
            llm_mbs,
            encoder_name,
            get_mock_data_seed(args, llm_grid, module_seed_offset=100_000),
        )
    if encoder_needs_data and llm_needs_data:
        return MockVLMIterator(
            args,
            llm_mbs,
            encoder_name,
            get_mock_data_seed(args, llm_grid, module_seed_offset=100_000),
        )
    return None


def build_schedule_pg_collection(
    encoder_name: str,
    encoder_grid: HyperCommGrid,
    llm_grid: HyperCommGrid,
    vision_pg: ProcessGroupCollection,
    language_pg: ProcessGroupCollection,
) -> MultiModuleProcessGroupCollection:
    """Build the schedule-facing process group collection for this rank."""
    module_pgs = {}
    language_model_module_name = None
    if is_rank_in_grid(encoder_grid):
        module_pgs[encoder_name] = vision_pg
    if is_rank_in_grid(llm_grid):
        module_pgs[MIMO_LANGUAGE_MODULE_KEY] = language_pg
        language_model_module_name = MIMO_LANGUAGE_MODULE_KEY

    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs, language_model_module_name=language_model_module_name
    )


def loss_func(loss_mask: Optional[torch.Tensor], output_tensor):
    """Return raw loss sum, local token count, and logging tensors."""
    if output_tensor is None:
        zero = torch.tensor(0.0, device="cuda", requires_grad=True)
        zero_count = torch.tensor(0, device="cuda", dtype=torch.int)
        return zero, zero_count, {"lm loss sum": zero.detach(), "lm tokens": zero_count}

    if isinstance(output_tensor, dict):
        output = output_tensor.get(
            MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
        )
    else:
        output = output_tensor

    if output is None:
        zero = torch.tensor(0.0, device="cuda", requires_grad=True)
        zero_count = torch.tensor(0, device="cuda", dtype=torch.int)
        return zero, zero_count, {"lm loss sum": zero.detach(), "lm tokens": zero_count}

    output = output.float()
    if loss_mask is None:
        raise RuntimeError("train_hetero.py requires a loss_mask for per-token loss")
    if output.shape != loss_mask.shape:
        raise RuntimeError(
            f"loss output shape {tuple(output.shape)} does not match loss_mask shape "
            f"{tuple(loss_mask.shape)}; per-token loss cannot be scaled correctly"
        )

    masked = output * loss_mask.float()
    num_tokens = loss_mask.float().sum().to(torch.int)
    loss_sum = masked.sum()
    return (
        loss_sum,
        num_tokens,
        {"lm loss sum": loss_sum.detach(), "lm tokens": num_tokens.detach()},
    )


def forward_step(data_iterator, model):
    """Forward step consumed by the MCore pipeline schedule."""
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    debug_rank("forward_step batch prepared")
    debug_rank("forward_step model call start")
    output_tensor, loss_mask = model(**batch)
    debug_rank("forward_step model call done")
    return output_tensor, partial(loss_func, loss_mask)


def get_global_batch_size(args: argparse.Namespace) -> int:
    """Return the language-side global batch size for scheduler accounting."""
    derived_global_batch_size = args.micro_batch_size * args.num_microbatches * args.llm_dp
    if args.global_batch_size is None:
        return derived_global_batch_size
    if args.global_batch_size != derived_global_batch_size:
        raise ValueError(
            "--global-batch-size must equal "
            "--micro-batch-size * --num-microbatches * --llm-dp in this hetero loop "
            f"({derived_global_batch_size}); got {args.global_batch_size}"
        )
    return args.global_batch_size


def build_optimizer_param_scheduler(args: argparse.Namespace, optimizer) -> OptimizerParamScheduler:
    """Build the MCore optimizer parameter scheduler using Megatron train-iters semantics."""
    global_batch_size = get_global_batch_size(args)
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.train_iters
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=args.lr,
        min_lr=args.min_lr if args.min_lr is not None else 0.0,
        lr_warmup_steps=args.lr_warmup_iters * global_batch_size,
        lr_decay_steps=lr_decay_iters * global_batch_size,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.weight_decay,
        end_wd=args.weight_decay,
        wd_incr_steps=args.train_iters * global_batch_size,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )


def run_train_loop(args: argparse.Namespace) -> None:
    """Run mock-data heterogeneous MIMO training."""
    apply_model_preset(args)
    apply_training_stage(args)
    resolve_image_token_id(args)
    world_size = dist.get_world_size()
    encoder_size, llm_size = validate_args(args, world_size)

    encoder_name = "images"
    debug_rank("creating encoder grid")
    encoder_grid = create_hypercomm_grid(
        offset=args.encoder_offset,
        tp=args.encoder_tp,
        cp=args.encoder_cp,
        pp=args.encoder_pp,
        dp=args.encoder_dp,
        ep=args.encoder_ep,
        expt_tp=args.encoder_expt_tp,
        expt_dp=args.encoder_expt_dp,
    )
    debug_rank("creating language grid")
    llm_grid = create_hypercomm_grid(
        offset=args.llm_offset,
        tp=args.llm_tp,
        cp=args.llm_cp,
        pp=args.llm_pp,
        dp=args.llm_dp,
        ep=args.llm_ep,
        expt_tp=args.llm_expt_tp,
        expt_dp=args.llm_expt_dp,
    )
    debug_rank("creating embedding groups")
    create_all_embedding_groups([encoder_grid, llm_grid])
    debug_rank("embedding groups ready")
    debug_rank("creating MIMO optimizer stats group")
    mimo_optimizer_stats_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    debug_rank("MIMO optimizer stats group ready")

    torch.manual_seed(args.seed)
    debug_rank("building MIMO model")
    mimo_model, module_to_grid_map, language_pg, vision_pg = build_mimo_model(
        args, encoder_grid, llm_grid, encoder_name
    )
    debug_rank("wiring training hooks")
    wire_training_hooks(mimo_model, language_pg, vision_pg, mimo_optimizer_stats_group)

    debug_rank("building MIMO optimizer")
    optimizer = get_mimo_optimizer(
        mimo_model,
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
        ),
        stats_group=mimo_optimizer_stats_group,
    )
    opt_param_scheduler = build_optimizer_param_scheduler(args, optimizer)
    debug_rank("MIMO optimizer ready")
    debug_rank("building pipeline communicator")
    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map,
        {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []},
        mimo_model.config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim={encoder_name: 2},
    )
    debug_rank("building schedule process groups")
    schedule_pg_collection = build_schedule_pg_collection(
        encoder_name, encoder_grid, llm_grid, vision_pg, language_pg
    )
    debug_rank("selecting data iterator")
    data_iterator = select_data_iterator(args, encoder_grid, llm_grid, encoder_name)
    debug_rank("training setup ready")

    print_rank_0(
        "Starting hetero MIMO mock training: "
        f"world_size={world_size}, encoder_size={encoder_size}, llm_size={llm_size}, "
        f"train_iters={args.train_iters}"
    )

    try:
        for iteration in range(1, args.train_iters + 1):
            zero_active_grad_buffers(mimo_model)
            optimizer.zero_grad()
            debug_rank(f"iteration {iteration}: starting forward/backward schedule")
            losses = schedule.forward_backward_pipelining_without_interleaving(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=[mimo_model],
                num_microbatches=args.num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                forward_only=False,
                p2p_communicator=communicator,
                pg_collection=schedule_pg_collection,
            )
            debug_rank(f"iteration {iteration}: schedule complete")
            debug_rank(f"iteration {iteration}: optimizer step starting")
            success, grad_norm, _ = optimizer.step()
            debug_rank(f"iteration {iteration}: optimizer step complete")
            if not success:
                raise RuntimeError(f"optimizer step failed at iteration {iteration}")
            opt_param_scheduler.step(increment=get_global_batch_size(args))

            if iteration % args.log_interval == 0:
                loss_acc = torch.zeros(2, dtype=torch.float32, device="cuda")
                is_log_stage = is_process_group_member(
                    getattr(language_pg, "tp_dp_cp", None)
                ) and is_pp_last_stage(language_pg.pp)
                if is_log_stage:
                    if losses and language_pg.tp.rank() == 0:
                        for loss_dict in losses:
                            loss_sum = loss_dict.get("lm loss sum")
                            num_tokens = loss_dict.get("lm tokens")
                            if isinstance(loss_sum, torch.Tensor):
                                loss_acc[0] += loss_sum.float()
                            elif loss_sum is not None:
                                loss_acc[0] += float(loss_sum)
                            if isinstance(num_tokens, torch.Tensor):
                                loss_acc[1] += num_tokens.float()
                            elif num_tokens is not None:
                                loss_acc[1] += float(num_tokens)
                    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM, group=language_pg.tp_dp_cp)
                    loss_value = (
                        loss_acc[0].item() / loss_acc[1].item() if loss_acc[1].item() else None
                    )
                    language_group_ranks = dist.get_process_group_ranks(language_pg.tp_dp_cp)
                    if dist.get_rank() == min(language_group_ranks):
                        sys.stdout.write(
                            f"iteration {iteration}: loss={loss_value}, grad_norm={grad_norm}\n"
                        )
                        sys.stdout.flush()
    finally:
        mimo_model.destroy()
        destroy_process_group_if_member(mimo_optimizer_stats_group)


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    initialize_distributed()
    try:
        run_train_loop(args)
        dist.barrier()
        print_rank_0("Heterogeneous MIMO mock training completed")
    finally:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        destroy_runtime_process_groups()
        parallel_state.destroy_global_memory_buffer()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
