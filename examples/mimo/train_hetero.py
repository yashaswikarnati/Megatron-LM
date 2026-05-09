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
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
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
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None


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
    model.add_argument("--hidden-size", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--vocab-size", type=int, default=512)
    model.add_argument("--seq-length", type=int, default=32)
    model.add_argument("--image-seq-length", type=int, default=None)
    model.add_argument("--image-token-id", type=int, default=50257)
    model.add_argument("--num-moe-experts", type=int, default=4)
    model.add_argument("--moe-router-topk", type=int, default=1)
    model.add_argument("--moe-grouped-gemm", action="store_true")
    model.add_argument(
        "--fp32", action="store_true", help="Build and train in fp32 instead of bf16"
    )

    train = parser.add_argument_group("training")
    train.add_argument("--micro-batch-size", type=int, default=2)
    train.add_argument("--num-microbatches", type=int, default=2)
    train.add_argument("--train-iters", type=int, default=2)
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--clip-grad", type=float, default=1.0)
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

    image_seq_length = args.image_seq_length or args.seq_length // 2
    if image_seq_length >= args.seq_length:
        raise ValueError("--image-seq-length must be smaller than --seq-length")
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


def projection_layer_spec() -> ModuleSpec:
    """Return the TE-backed projection MLP spec."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def language_model_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    llm_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the GPT language ModuleSpec for the local language grid."""
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
    """Create the mock vision ModuleSpec for the local encoder grid."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    tp_size = get_group_size_or(tp_pg, get_grid_dim_size(encoder_grid, "tp"))
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(encoder_grid, "pp"))
    pp_rank = get_group_rank_or(pp_pg)
    bf16 = not args.fp32

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
    mimo_model = MimoModel(mimo_config)
    debug_rank("moving MimoModel to cuda")
    mimo_model.to(torch.device("cuda"))
    if not args.fp32:
        mimo_model.to(torch.bfloat16)
    debug_rank("MimoModel moved to target dtype/device")

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, bucket_size=10000, use_distributed_optimizer=True
    )
    if mimo_model.language_model is not None:
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
            debug_rank("wrapping vision submodule in DDP")
            mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                config=submodule.encoders["clip_encoder"].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=vision_pg,
            )
            debug_rank("vision submodule DDP ready")

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
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = input_ids.clone()
        labels[input_ids == args.image_token_id] = -100

        loss_mask = torch.ones(
            self.micro_batch_size, args.seq_length, dtype=torch.float32, device="cuda"
        )
        loss_mask[input_ids == args.image_token_id] = 0.0

        encoder_hidden_states = torch.randn(
            self.image_seq_length,
            self.micro_batch_size,
            args.hidden_size,
            device="cuda",
            dtype=self.dtype,
            generator=self.generator,
        )
        num_image_placeholders = (input_ids == args.image_token_id).sum().item()
        expected_image_placeholders = self.image_seq_length * self.micro_batch_size
        if num_image_placeholders != expected_image_placeholders:
            raise RuntimeError(
                f"mock batch has {num_image_placeholders} image placeholders, "
                f"expected {expected_image_placeholders}"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(args.seq_length, device="cuda")
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone(),
            "modality_inputs": {
                self.encoder_name: {
                    "clip_encoder": {"hidden_states": encoder_hidden_states, "attention_mask": None}
                }
            },
        }


def wire_training_hooks(
    mimo_model: MimoModel, language_pg: ProcessGroupCollection, vision_pg: ProcessGroupCollection
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

        token_count = num_tokens.to(device="cuda", dtype=torch.float32)
        if not is_token_source_rank():
            token_count.zero_()
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        global_num_tokens = token_count.item()

        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=vision_pg,
                    force_all_reduce=force_all_reduce,
                )

        if global_num_tokens > 0:
            scale = 1.0 / global_num_tokens
            if mimo_model.language_model is not None:
                mimo_model.language_model.scale_gradients(scale)
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
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
    if loss_mask is not None and output.shape == loss_mask.shape:
        masked = output * loss_mask.float()
        num_tokens = loss_mask.float().sum().to(torch.int)
        loss_sum = masked.sum()
    else:
        loss_sum = output.sum()
        num_tokens = torch.tensor(output.numel(), device="cuda", dtype=torch.int)
    return (
        loss_sum,
        num_tokens,
        {"lm loss sum": loss_sum.detach(), "lm tokens": num_tokens.detach()},
    )


def forward_step(data_iterator, model):
    """Forward step consumed by the MCore pipeline schedule."""
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask)


def run_train_loop(args: argparse.Namespace) -> None:
    """Run mock-data heterogeneous MIMO training."""
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

    torch.manual_seed(args.seed + dist.get_rank())
    debug_rank("building MIMO model")
    mimo_model, module_to_grid_map, language_pg, vision_pg = build_mimo_model(
        args, encoder_grid, llm_grid, encoder_name
    )
    debug_rank("wiring training hooks")
    wire_training_hooks(mimo_model, language_pg, vision_pg)

    debug_rank("building MIMO optimizer")
    optimizer = get_mimo_optimizer(
        mimo_model,
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
        ),
    )
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
            optimizer.zero_grad()
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
            success, grad_norm, _ = optimizer.step()
            if not success:
                raise RuntimeError(f"optimizer step failed at iteration {iteration}")

            if iteration % args.log_interval == 0:
                loss_acc = torch.zeros(2, dtype=torch.float32, device="cuda")
                if (
                    losses
                    and is_process_group_member(getattr(language_pg, "pp", None))
                    and is_process_group_member(getattr(language_pg, "tp", None))
                ):
                    is_log_source = is_pp_last_stage(language_pg.pp) and language_pg.tp.rank() == 0
                    if is_log_source:
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
                dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
                loss_value = loss_acc[0].item() / loss_acc[1].item() if loss_acc[1].item() else None
                if dist.get_rank() == 0:
                    print_rank_0(f"iteration {iteration}: loss={loss_value}, grad_norm={grad_norm}")
    finally:
        mimo_model.destroy()


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
            dist.barrier()
        except Exception:
            pass
        destroy_runtime_process_groups()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
