# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generate artifacts for homogeneous-vs-heterogeneous MIMO training parity.

The runner intentionally lives outside ``examples/mimo/train.py`` and
``examples/mimo/train_hetero.py``. It exercises the same runtime pieces while
using a deterministic sample stream and an explicit initial checkpoint so the
only variable under test is the training-loop orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    language_model_spec,
    prepare_model_provider_args,
    validate_model_provider_args,
    vision_submodules_spec,
)
from examples.mimo.training.hetero.args import prepare_args as prepare_hetero_args
from examples.mimo.training.hetero.grad_sync import configure_grad_sync, zero_active_grad_buffers
from examples.mimo.training.hetero.loop import build_pipeline_communicator
from examples.mimo.training.hetero.optimizer import (
    build_optimizer,
    build_optimizer_param_scheduler,
    get_global_batch_size,
)
from examples.mimo.training.hetero.runtime import build_mimo_runtime
from examples.mimo.training.hetero.step import loss_func, move_batch_to_cuda, reduce_update_success
from examples.mimo.training.hetero.topology import (
    ENCODER_MODULE_NAME,
    HeteroTopology,
    create_topology,
    get_grid_coordinate,
    is_rank_in_grid,
)
from examples.mimo.utils.hetero import is_process_group_member
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_model_config, unwrap_model


@dataclass
class GridShape:
    """Minimal grid shape object consumed by the shared model-provider helpers."""

    shape: list[int]
    dim_names: list[str]


class DeterministicVLMIterator:
    """Iterator that maps DP lanes to deterministic global sample ids."""

    def __init__(
        self,
        args: argparse.Namespace,
        micro_batch_size: int,
        dp_rank: int,
        llm_dp: int,
        encoder_name: str,
    ) -> None:
        self.args = args
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.llm_dp = llm_dp
        self.encoder_name = encoder_name
        self.microbatch_index = 0
        self.consumed_sample_ids: list[int] = []
        self.image_seq_length = args.image_seq_length or args.seq_length // 2
        allowed = [
            token_id
            for token_id in range(1, args.vocab_size)
            if token_id not in {args.pad_token_id, args.image_token_id}
        ]
        if not allowed:
            raise RuntimeError("validation data requires at least one non-special token")
        self.allowed_tokens = torch.tensor(allowed, dtype=torch.long, device="cuda")

    def __iter__(self):
        return self

    def __next__(self):
        global_micro_batch = self.args.micro_batch_size * self.llm_dp
        start = self.microbatch_index * global_micro_batch + self.dp_rank * self.micro_batch_size
        sample_ids = list(range(start, start + self.micro_batch_size))
        self.microbatch_index += 1
        self.consumed_sample_ids.extend(sample_ids)
        return self._batch_for_sample_ids(sample_ids)

    def drain_sample_ids(self) -> list[int]:
        sample_ids = self.consumed_sample_ids
        self.consumed_sample_ids = []
        return sample_ids

    def _batch_for_sample_ids(self, sample_ids: list[int]) -> dict[str, Any]:
        args = self.args
        batch_size = len(sample_ids)
        text_length = args.seq_length - self.image_seq_length
        if text_length < 2:
            raise RuntimeError("validation data requires at least two text tokens")

        sample_tensor = torch.tensor(sample_ids, dtype=torch.long, device="cuda").view(-1, 1)
        text_pos = torch.arange(text_length, dtype=torch.long, device="cuda").view(1, -1)
        token_indices = (sample_tensor * 131 + text_pos * 17 + 7) % self.allowed_tokens.numel()
        text_tokens = self.allowed_tokens[token_indices]
        image_tokens = torch.full(
            (batch_size, self.image_seq_length),
            args.image_token_id,
            dtype=torch.long,
            device="cuda",
        )
        input_ids = torch.cat((image_tokens, text_tokens), dim=1)

        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        labels[(labels == args.image_token_id) | (labels == args.pad_token_id)] = -100
        loss_mask = (labels != -100).to(torch.float32)

        dtype = torch.float32 if args.fp32 else torch.bfloat16
        seq = torch.arange(self.image_seq_length, dtype=torch.float32, device="cuda").view(-1, 1, 1)
        batch = torch.tensor(sample_ids, dtype=torch.float32, device="cuda").view(1, -1, 1)
        hidden = torch.arange(args.hidden_size, dtype=torch.float32, device="cuda").view(1, 1, -1)
        hidden_states = torch.sin(batch * 0.013 + seq * 0.017 + hidden * 0.001).to(dtype=dtype)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(args.seq_length, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone(),
            "modality_inputs": {
                self.encoder_name: {
                    args.vision_encoder_key: {
                        "hidden_states": hidden_states,
                        "attention_mask": None,
                    }
                }
            },
        }


def parse_args() -> argparse.Namespace:
    """Parse validation-runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["init", "homo", "hetero"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-state-path", type=Path, default=None)
    parser.add_argument("--snapshot-interval", type=int, default=1)

    grid = parser.add_argument_group("module grids")
    grid.add_argument("--encoder-offset", type=int, default=0)
    grid.add_argument("--encoder-tp", type=int, default=1)
    grid.add_argument("--encoder-cp", type=int, default=1)
    grid.add_argument("--encoder-pp", type=int, default=1)
    grid.add_argument("--encoder-dp", type=int, default=1)
    grid.add_argument("--encoder-ep", type=int, default=1)
    grid.add_argument("--encoder-expt-tp", type=int, default=None)
    grid.add_argument("--encoder-expt-dp", type=int, default=None)
    grid.add_argument("--llm-offset", type=int, default=1)
    grid.add_argument("--llm-tp", type=int, default=1)
    grid.add_argument("--llm-cp", type=int, default=1)
    grid.add_argument("--llm-pp", type=int, default=1)
    grid.add_argument("--llm-dp", type=int, default=1)
    grid.add_argument("--llm-ep", type=int, default=1)
    grid.add_argument("--llm-expt-tp", type=int, default=1)
    grid.add_argument("--llm-expt-dp", type=int, default=1)

    add_model_provider_args(parser)

    train = parser.add_argument_group("training")
    train.add_argument("--dataset-provider", default="mock")
    train.add_argument("--micro-batch-size", type=int, default=1)
    train.add_argument("--global-batch-size", type=int, default=None)
    train.add_argument("--num-microbatches", type=int, default=2)
    train.add_argument("--train-iters", type=int, default=4)
    train.add_argument("--lr", type=float, default=2.0e-4)
    train.add_argument("--min-lr", type=float, default=0.0)
    train.add_argument("--lr-decay-style", type=str, default="constant")
    train.add_argument("--lr-warmup-iters", type=int, default=0)
    train.add_argument("--lr-decay-iters", type=int, default=None)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--log-num-zeros-in-grad", action="store_true")
    train.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument("--ddp-bucket-size", type=int, default=0)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.initial_state_path is not None:
        args.initial_state_path = args.initial_state_path.resolve()
    return args


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    configure_validation_math()
    if args.mode == "hetero":
        run_hetero(args)
    else:
        run_homogeneous_or_init(args)


def configure_validation_math() -> None:
    """Use stable fp32 math for cross-layout parity comparisons."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except AttributeError:
        pass


def run_homogeneous_or_init(args: argparse.Namespace) -> None:
    """Run the colocated homogeneous baseline or emit its initial checkpoint."""
    initialize_torch_distributed()
    try:
        prepare_colocated_args(args)
        initialize_parallel_state(args)
        torch.manual_seed(args.seed)
        model_parallel_cuda_manual_seed(args.seed, force_reset_rng=True)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = build_colocated_model(args, pg_collection)
        try:
            if args.mode == "init":
                save_initial_state(args, model)
                return
            if args.initial_state_path is None:
                raise ValueError("--initial-state-path is required for --mode homo")
            load_initial_state(args.initial_state_path, model)
            disable_stochastic_layers_for_parity(model)
            ddp_model = wrap_colocated_model(args, model, pg_collection)
            optimizer = build_colocated_optimizer(args, ddp_model, pg_collection)
            opt_param_scheduler = build_optimizer_param_scheduler(args, optimizer)
            configure_colocated_schedule(ddp_model, optimizer, pg_collection)
            data_iterator = build_homogeneous_iterator(args, pg_collection)
            run_training_steps(
                args=args,
                model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                data_iterator=data_iterator,
                pg_collection=pg_collection,
                topology=None,
            )
        finally:
            model.destroy()
    finally:
        shutdown_distributed(destroy_parallel_state=True)


def run_hetero(args: argparse.Namespace) -> None:
    """Run the standalone heterogeneous path with validation data."""
    initialize_torch_distributed()
    try:
        ensure_global_memory_buffer()
        world_size = dist.get_world_size()
        encoder_size, llm_size = prepare_hetero_args(args, world_size)
        torch.manual_seed(args.seed)
        topology: Optional[HeteroTopology] = None
        model: Optional[MimoModel] = None
        try:
            topology = create_topology(args, encoder_size, llm_size)
            model = build_mimo_runtime(args, topology)
            if args.initial_state_path is None:
                raise ValueError("--initial-state-path is required for --mode hetero")
            load_initial_state(args.initial_state_path, model, topology=topology)
            disable_stochastic_layers_for_parity(model)
            configure_grad_sync(model, topology)
            optimizer = build_optimizer(args, model)
            opt_param_scheduler = build_optimizer_param_scheduler(args, optimizer)
            communicator = build_pipeline_communicator(model, topology)
            data_iterator = build_hetero_iterator(args, topology)
            run_training_steps(
                args=args,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                data_iterator=data_iterator,
                pg_collection=topology.schedule_pg_collection,
                topology=topology,
                communicator=communicator,
            )
        finally:
            if model is not None:
                model.destroy()
            if topology is not None:
                topology.destroy()
    finally:
        shutdown_distributed(destroy_parallel_state=False)


def initialize_torch_distributed() -> None:
    """Initialize torch.distributed for a torchrun-launched validation process."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    dist.barrier()


def ensure_global_memory_buffer() -> None:
    """Initialize Megatron's global memory buffer when no parallel_state init will do it."""
    try:
        parallel_state.get_global_memory_buffer()
    except AssertionError:
        parallel_state._set_global_memory_buffer()


def initialize_parallel_state(args: argparse.Namespace) -> None:
    """Initialize Megatron global parallel_state for the homogeneous baseline."""
    expected_world_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    if dist.get_world_size() != expected_world_size:
        raise ValueError(
            f"homogeneous world_size must equal llm_tp*llm_cp*llm_pp*llm_dp "
            f"({expected_world_size}); got {dist.get_world_size()}"
        )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.llm_tp,
        pipeline_model_parallel_size=args.llm_pp,
        context_parallel_size=args.llm_cp,
        expert_model_parallel_size=args.llm_ep,
        expert_tensor_parallel_size=args.llm_expt_tp,
        create_gloo_process_groups=False,
        order="tp-cp-ep-dp-pp",
    )


def shutdown_distributed(*, destroy_parallel_state: bool) -> None:
    """Destroy distributed state."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    if destroy_parallel_state:
        parallel_state.destroy_model_parallel()
    parallel_state.destroy_global_memory_buffer()
    if dist.is_initialized():
        dist.destroy_process_group()


def prepare_colocated_args(args: argparse.Namespace) -> None:
    """Apply provider defaults and validation for the homogeneous baseline."""
    prepare_model_provider_args(args)
    validate_model_provider_args(args)
    if args.dataset_provider != "mock":
        raise ValueError("training parity currently uses deterministic mock data only")
    if args.llm_cp != 1:
        raise ValueError("training parity currently supports CP=1 only")
    if args.global_batch_size is None:
        args.global_batch_size = get_global_batch_size(args)


def build_colocated_model(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection
) -> MimoModel:
    """Build a colocated MIMO model using the shared current provider pieces."""
    grid = GridShape(
        shape=[args.llm_tp, args.llm_cp, args.llm_dp, args.llm_pp],
        dim_names=["tp", "cp", "dp", "pp"],
    )
    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec(args, pg_collection, grid),
        modality_submodules_spec={
            ENCODER_MODULE_NAME: vision_submodules_spec(args, pg_collection, grid)
        },
        special_token_ids={ENCODER_MODULE_NAME: args.image_token_id},
    )
    model = MimoModel(mimo_config, cp_group=pg_collection.cp, tp_group=pg_collection.tp)
    model.cuda(torch.cuda.current_device())
    if not args.fp32:
        model.to(torch.bfloat16)
    return model


def wrap_colocated_model(
    args: argparse.Namespace, model: MimoModel, pg_collection: ProcessGroupCollection
) -> DistributedDataParallel:
    """Wrap the colocated baseline in the same MCore DDP type used by Megatron training."""
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=args.overlap_grad_reduce,
        bucket_size=args.ddp_bucket_size if args.ddp_bucket_size > 0 else None,
        use_distributed_optimizer=True,
    )
    return DistributedDataParallel(
        config=model.config,
        ddp_config=ddp_config,
        module=model,
        pg_collection=pg_collection,
    )


def build_colocated_optimizer(
    args: argparse.Namespace,
    model: DistributedDataParallel,
    pg_collection: ProcessGroupCollection,
):
    """Build the homogeneous Megatron optimizer."""
    return get_megatron_optimizer(
        config=OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
        model_chunks=[model],
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )


def configure_colocated_schedule(
    model: DistributedDataParallel, optimizer, pg_collection: ProcessGroupCollection
) -> None:
    """Attach Megatron training-loop callbacks to the colocated model config."""
    config = get_model_config(model)
    config.grad_scale_func = optimizer.scale_loss
    config.finalize_model_grads_func = finalize_model_grads
    if model.ddp_config.overlap_grad_reduce:
        config.no_sync_func = model.no_sync


def build_homogeneous_iterator(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection
) -> Optional[DeterministicVLMIterator]:
    """Create the deterministic data iterator for homogeneous data ranks."""
    needs_data = is_pp_first_stage(pg_collection.pp) or is_pp_last_stage(pg_collection.pp)
    if not needs_data:
        return None
    dp_rank = dist.get_rank(pg_collection.dp)
    return DeterministicVLMIterator(
        args=args,
        micro_batch_size=args.micro_batch_size,
        dp_rank=dp_rank,
        llm_dp=args.llm_dp,
        encoder_name=ENCODER_MODULE_NAME,
    )


def build_hetero_iterator(
    args: argparse.Namespace, topology: HeteroTopology
) -> Optional[DeterministicVLMIterator]:
    """Create deterministic role-local iterators for a hetero run."""
    encoder_grid = topology.encoder_grid
    llm_grid = topology.llm_grid
    encoder_needs_data = is_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )
    if not encoder_needs_data and not llm_needs_data:
        return None

    if encoder_needs_data and not llm_needs_data:
        if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
            raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")
        encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp
        encoder_dp_rank = get_grid_coordinate(encoder_grid, "dp")
        return DeterministicVLMIterator(
            args=args,
            micro_batch_size=encoder_mbs,
            dp_rank=encoder_dp_rank,
            llm_dp=args.llm_dp,
            encoder_name=topology.encoder_name,
        )

    llm_dp_rank = get_grid_coordinate(llm_grid, "dp")
    return DeterministicVLMIterator(
        args=args,
        micro_batch_size=args.micro_batch_size,
        dp_rank=llm_dp_rank,
        llm_dp=args.llm_dp,
        encoder_name=topology.encoder_name,
    )


def validation_forward_step(data_iterator, model):
    """Forward step used by both parity paths."""
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    batch = move_batch_to_cuda(batch)
    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask=loss_mask)


def disable_stochastic_layers_for_parity(model: torch.nn.Module) -> None:
    """Remove dropout randomness while keeping modules in train mode for optimizer parity."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for attr in ("attention_dropout", "hidden_dropout"):
            value = getattr(module, attr, None)
            if isinstance(value, (float, int)):
                setattr(module, attr, 0.0)
        config = getattr(module, "config", None)
        if config is None:
            continue
        for attr in ("attention_dropout", "hidden_dropout"):
            if hasattr(config, attr):
                setattr(config, attr, 0.0)


def run_training_steps(
    *,
    args: argparse.Namespace,
    model,
    optimizer,
    opt_param_scheduler,
    data_iterator: Optional[DeterministicVLMIterator],
    pg_collection,
    topology: Optional[HeteroTopology],
    communicator=None,
) -> None:
    """Run training and emit per-rank parity artifacts."""
    metrics_path = args.output_dir / args.mode / f"metrics_rank_{dist.get_rank():05d}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.train_iters + 1):
        if topology is None:
            model.zero_grad_buffer()
        else:
            zero_active_grad_buffers(model)
        optimizer.zero_grad()

        schedule_pg_collection = pg_collection if communicator is not None else None
        losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=validation_forward_step,
            data_iterator=data_iterator,
            model=[model],
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
            p2p_communicator=communicator,
            pg_collection=schedule_pg_collection,
        )
        grad_snapshot = collect_state_entries(model, optimizer, include_params=False)

        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        update_successful = reduce_update_success(update_successful)
        skipped_iter = 0
        if update_successful:
            opt_param_scheduler.step(increment=get_global_batch_size(args))
        else:
            skipped_iter = 1

        state_snapshot = collect_state_entries(model, optimizer, include_params=True)
        state_snapshot["grads"] = grad_snapshot["grads"]
        if should_snapshot(args, iteration):
            write_snapshot(args, iteration, state_snapshot)

        loss_value = reduce_loss_for_mode(losses, topology)
        sample_ids = data_iterator.drain_sample_ids() if data_iterator is not None else []
        append_metric(
            metrics_path,
            {
                "iteration": iteration,
                "rank": dist.get_rank(),
                "loss": loss_value,
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "num_zeros_in_grad": int(num_zeros_in_grad)
                if num_zeros_in_grad is not None
                else None,
                "lr": get_canonical_lr_for_logging(optimizer.param_groups),
                "skipped_iter": skipped_iter,
                "update_successful": bool(update_successful),
                "sample_ids": sample_ids,
            },
        )
        dist.barrier()


def should_snapshot(args: argparse.Namespace, iteration: int) -> bool:
    """Return whether this iteration should include state artifacts."""
    return args.snapshot_interval > 0 and iteration % args.snapshot_interval == 0


@torch.no_grad()
def reduce_loss_for_mode(losses: list[dict[str, Any]], topology: Optional[HeteroTopology]):
    """Return a scalar reduced LM loss on the logging rank, else None."""
    if topology is not None:
        language_pg = topology.language_pg
        is_log_stage = (
            is_process_group_member(getattr(language_pg, "dp_cp", None))
            and is_pp_last_stage(language_pg.pp)
            and language_pg.tp.rank() == 0
        )
        if not is_log_stage:
            return None
        loss_acc = sum_loss_vectors(losses)
        dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM, group=language_pg.dp_cp)
        if dist.get_rank(language_pg.dp_cp) != 0:
            return None
        return loss_acc[0].item() / loss_acc[1].item() if loss_acc[1].item() else None

    if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        return None
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return None
    loss_acc = sum_loss_vectors(losses)
    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM, group=parallel_state.get_data_parallel_group())
    if parallel_state.get_data_parallel_rank() != 0:
        return None
    return loss_acc[0].item() / loss_acc[1].item() if loss_acc[1].item() else None


def sum_loss_vectors(losses: list[dict[str, Any]]) -> torch.Tensor:
    """Sum Megatron two-element loss vectors from all microbatches."""
    loss_acc = torch.zeros(2, dtype=torch.float32, device="cuda")
    for loss_dict in losses:
        loss = loss_dict.get("lm loss")
        if isinstance(loss, torch.Tensor):
            loss_acc += loss.detach().to(device="cuda", dtype=torch.float32).view(2)
        elif loss is not None:
            loss_acc += torch.tensor(loss, dtype=torch.float32, device="cuda").view(2)
    return loss_acc


def append_metric(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON metric record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_snapshot(args: argparse.Namespace, iteration: int, snapshot: dict[str, Any]) -> None:
    """Write one per-rank state snapshot."""
    snapshot_dir = args.output_dir / args.mode / f"iter_{iteration:06d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, snapshot_dir / f"rank_{dist.get_rank():05d}.pt")


def collect_state_entries(model, optimizer, *, include_params: bool) -> dict[str, Any]:
    """Collect named params, grads, and optimizer-state shards for this rank."""
    param_to_name, param_shapes = named_model_parameters(model)
    param_ranges, has_distributed_ranges = optimizer_param_ranges(optimizer, param_to_name)
    snapshot = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "params": {},
        "grads": {},
        "optimizer": {},
    }

    for param, name in param_to_name.items():
        shape = param_shapes[name]
        if include_params:
            snapshot["params"][name] = tensor_entry(
                param.detach(), shape=shape, value_range=(0, param.numel())
            )
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is not None:
            value_range = param_ranges.get(name)
            if value_range is None:
                if has_distributed_ranges:
                    continue
                value_range = (0, param.numel())
            grad_to_save = grad.detach()
            if grad.numel() == param.numel():
                if value_range != (0, param.numel()):
                    start, end = value_range
                    grad_to_save = grad_to_save.view(-1)[start:end]
                else:
                    value_range = (0, param.numel())
            snapshot["grads"][name] = tensor_entry(
                grad_to_save, shape=shape, value_range=value_range
            )

    if include_params:
        snapshot["optimizer"] = collect_optimizer_entries(optimizer, param_to_name, param_shapes)
    return snapshot


def named_model_parameters(model) -> tuple[dict[torch.nn.Parameter, str], dict[str, tuple[int, ...]]]:
    """Return a normalized parameter-object to logical-name mapping."""
    param_to_name: dict[torch.nn.Parameter, str] = {}
    param_shapes: dict[str, tuple[int, ...]] = {}
    for name, param in model.named_parameters():
        logical_name = normalize_param_name(name)
        param_to_name[param] = logical_name
        param_shapes[logical_name] = tuple(param.shape)
    return param_to_name, param_shapes


def normalize_param_name(name: str) -> str:
    """Normalize DDP wrapper segments out of parameter names."""
    if name.startswith("module."):
        name = name[len("module.") :]
    return name.replace(".module.", ".")


def optimizer_param_ranges(
    optimizer, param_to_name: dict[torch.nn.Parameter, str]
) -> tuple[dict[str, tuple[int, int]], bool]:
    """Return local distributed-optimizer ranges keyed by logical parameter name."""
    ranges: dict[str, tuple[int, int]] = {}
    has_distributed_ranges = False
    for leaf in iter_leaf_optimizers(optimizer):
        if not hasattr(leaf, "model_chunks") or not hasattr(leaf, "_get_model_param_range_map"):
            continue
        has_distributed_ranges = True
        for chunk in leaf.model_chunks:
            for _, param in chunk.named_parameters():
                if param not in param_to_name:
                    continue
                try:
                    range_map = leaf._get_model_param_range_map(param)
                except Exception:
                    continue
                param_range = range_map["param"]
                ranges[param_to_name[param]] = (int(param_range.start), int(param_range.end))
    return ranges, has_distributed_ranges


def collect_optimizer_entries(
    optimizer,
    param_to_name: dict[torch.nn.Parameter, str],
    param_shapes: dict[str, tuple[int, ...]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect local optimizer state tensors keyed by logical parameter name."""
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    for leaf in iter_leaf_optimizers(optimizer):
        if not hasattr(leaf, "model_chunks") or not hasattr(
            leaf, "_get_main_param_and_optimizer_states"
        ):
            continue
        for chunk in leaf.model_chunks:
            for _, param in chunk.named_parameters():
                if param not in param_to_name:
                    continue
                name = param_to_name[param]
                try:
                    state = leaf._get_main_param_and_optimizer_states(param)
                except Exception:
                    continue
                value_range = (0, param.numel())
                if hasattr(leaf, "_get_model_param_range_map"):
                    range_map = leaf._get_model_param_range_map(param)
                    param_range = range_map["param"]
                    value_range = (int(param_range.start), int(param_range.end))
                param_entries = entries.setdefault(name, {})
                for state_key, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.numel() == param.numel():
                        state_range = (0, param.numel())
                        shape = param_shapes[name]
                    elif tensor.ndim == 0:
                        state_range = (0, 1)
                        shape = ()
                    else:
                        state_range = value_range
                        shape = param_shapes[name]
                    param_entries[state_key] = tensor_entry(
                        tensor.detach(), shape=shape, value_range=state_range
                    )
    return entries


def iter_leaf_optimizers(optimizer) -> Iterable[Any]:
    """Yield leaf Megatron optimizers from MIMO or chained wrappers."""
    module_infos = getattr(optimizer, "module_infos", None)
    if module_infos is not None:
        for info in module_infos.values():
            if info.optimizer is not None:
                yield from iter_leaf_optimizers(info.optimizer)
        return
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for child in chained:
            yield from iter_leaf_optimizers(child)
        return
    yield optimizer


def tensor_entry(
    tensor: torch.Tensor, *, shape: tuple[int, ...], value_range: tuple[int, int]
) -> dict[str, Any]:
    """Serialize a tensor shard with enough metadata to reconstruct logical tensors."""
    return {
        "tensor": tensor.detach().float().cpu().contiguous(),
        "shape": tuple(int(dim) for dim in shape),
        "range": tuple(int(v) for v in value_range),
        "numel": int(torch.tensor(shape).prod().item()) if shape else 1,
    }


def save_initial_state(args: argparse.Namespace, model: MimoModel) -> None:
    """Save an explicit colocated initial checkpoint for both validation paths."""
    if dist.get_rank() != 0:
        return
    if args.initial_state_path is None:
        args.initial_state_path = args.output_dir / "initial_state.pt"
    args.initial_state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if model.language_model is not None:
        state["language_model"] = module_state_to_cpu(unwrap_model(model.language_model))
    state["modality_submodules"] = {}
    for name, submodule in model.modality_submodules.items():
        state["modality_submodules"][name] = module_state_to_cpu(unwrap_model(submodule))
    torch.save(state, args.initial_state_path)
    append_metric(
        args.output_dir / args.mode / "metrics_rank_00000.jsonl",
        {"initial_state_path": str(args.initial_state_path)},
    )


def module_state_to_cpu(module: torch.nn.Module) -> dict[str, Any]:
    """Return a CPU-cloned state dict."""
    state = {}
    for name, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[name] = value.detach().cpu().clone()
        else:
            state[name] = value
    return state


def load_initial_state(
    path: Path, model: MimoModel, *, topology: Optional[HeteroTopology] = None
) -> None:
    """Load an explicit initial checkpoint into active local modules."""
    state = torch.load(path, map_location="cpu")
    if model.language_model is not None:
        load_module_state(unwrap_model(model.language_model), state["language_model"])
    for name, submodule_state in state["modality_submodules"].items():
        if name in model.modality_submodules:
            submodule = model.modality_submodules[name]
            if submodule is not None:
                load_module_state(unwrap_model(submodule), submodule_state)
    dist.barrier()


def load_module_state(module: torch.nn.Module, state: dict[str, Any]) -> None:
    """Load a CPU checkpoint shard into a CUDA module."""
    cuda_state = {
        name: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in state.items()
    }
    incompatible = module.load_state_dict(cuda_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"state load mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


if __name__ == "__main__":
    main()
