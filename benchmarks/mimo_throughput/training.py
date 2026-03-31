# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unified training loop for MIMO throughput benchmarking.

Supports both colocated heterogeneous parallelism (encoder TP/DP != LLM TP/DP)
and homogeneous configurations (encoder TP/DP == LLM TP/DP), with PP=1 or PP>1.

Model creation, DDP wrapping, optimizer setup, and the forward-backward loop
are all handled here. The entry point is ``run_benchmark(config)``.
"""

import gc
import logging
import os
import time
from contextlib import ExitStack, contextmanager
from functools import partial

# Ensure TP all-reduce overlaps with backward compute (TE recommendation)
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import torch
import torch.distributed as dist

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
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from benchmarks.mimo_throughput.config import BenchmarkConfig
from benchmarks.mimo_throughput.data import SyntheticVLMIterator
from benchmarks.mimo_throughput.metrics import PerformanceMonitor
from benchmarks.mimo_throughput.process_groups import ProcessGroupManager

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None

logger = logging.getLogger(__name__)

ENCODER_NAME = "images"


# ---------------------------------------------------------------------------
# Distributed optimizer instance group helpers
# ---------------------------------------------------------------------------


def _create_dist_opt_instance_groups(grid, num_instances):
    """Create hierarchical distributed optimizer instance groups from a HyperCommGrid.

    Mirrors the logic in ``parallel_state._initialize_expert_data_parallel_groups``
    (lines 1291-1313) but operates on a HyperCommGrid instead of global state.

    Args:
        grid: HyperCommGrid with at least a ``dp`` dimension.
        num_instances: Number of distributed optimizer instances (must divide dp_size).

    Returns:
        (intra_dp_group, inter_group, intra_dist_opt_group) for the calling rank.
    """
    from megatron.core.parallel_state import create_hierarchical_groups

    rank = dist.get_rank()
    dp_size = grid.shape[grid.dim_names.index('dp')]
    assert dp_size % num_instances == 0
    intra_size = dp_size // num_instances

    # Intra/inter DP groups — one set per TP group (matches parallel_state pattern).
    intra_dp_group = None
    inter_group = None
    for dp_ranks in grid.get_rank_enum('dp'):
        hierarchical_groups, _ = create_hierarchical_groups(
            rank, dp_ranks, [intra_size, num_instances], group_desc="DIST_OPT_INSTANCE",
        )
        if rank in dp_ranks:
            intra_dp_group = hierarchical_groups[0]
            inter_group = hierarchical_groups[1]

    # intra_dist_opt: spans all model-parallel dims × intra-DP (for grad-stats).
    # Build by collecting, for each intra-DP slice, all MP peers of those DP ranks.
    # This is order-independent — get_rank_enum returns groups whose membership is
    # determined by the grid shape, not by dim_names ordering.
    tp_rank_groups = grid.get_rank_enum('tp')
    rank_to_mp_peers = {}
    for tp_ranks in tp_rank_groups:
        for r in tp_ranks:
            rank_to_mp_peers[r] = tp_ranks

    intra_dist_opt_group = None
    seen = set()
    representative_dp_ranks = grid.get_rank_enum('dp')[0]
    for start in range(0, len(representative_dp_ranks), intra_size):
        intra_dp_ranks = representative_dp_ranks[start : start + intra_size]
        all_ranks = sorted({r for dp_r in intra_dp_ranks for r in rank_to_mp_peers[dp_r]})
        key = tuple(all_ranks)
        if key not in seen:
            seen.add(key)
            group = dist.new_group(ranks=all_ranks)
            if rank in all_ranks:
                intra_dist_opt_group = group

    return intra_dp_group, inter_group, intra_dist_opt_group


# ---------------------------------------------------------------------------
# Model spec helpers (mirrors test_mimo_colocated_e2e.py patterns)
# ---------------------------------------------------------------------------


def _get_language_model_spec(arch, pg_collection):
    """Build ModuleSpec for the GPT language model."""
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    lm_config = TransformerConfig(
        num_layers=arch.num_layers,
        hidden_size=arch.hidden_size,
        num_attention_heads=arch.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
        gradient_accumulation_fusion=True,
        bias_dropout_fusion=True,
        bias_activation_fusion=True,
        sequence_parallel=tp_size > 1,
        tp_comm_overlap=tp_size > 1,
        tp_comm_overlap_rs_dgrad=False,
        tp_comm_overlap_rs=False,
    )
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": arch.vocab_size,
            "max_sequence_length": arch.seq_length,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": pg_collection,
            # Disable SP scatter in embedding — MIMO's align_embeddings needs the
            # full [S, B, H] sequence. We scatter to SP region after alignment.
            "scatter_embedding_sequence_parallel": False,
        },
    )


def _get_vision_submodules_spec(arch, language_hidden_size, pg_collection):
    """Build ModuleSpec for the vision encoder + projection."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    vision_config = TransformerConfig(
        num_layers=arch.num_layers,
        hidden_size=arch.hidden_size,
        num_attention_heads=arch.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        recompute_granularity='full',
        recompute_method='uniform',
        recompute_num_layers=arch.num_layers,
    )

    proj_cfg = TransformerConfig(
        num_layers=1, hidden_size=language_hidden_size, num_attention_heads=1
    )
    proj_cfg.ffn_hidden_size = language_hidden_size
    proj_cfg.bias_activation_fusion = True
    proj_cfg.add_bias_linear = True
    proj_cfg.activation_func = torch.nn.functional.gelu

    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")

    return ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {
                "clip_encoder": ModuleSpec(
                    module=TransformerBlock,
                    params={
                        "config": vision_config,
                        "spec": get_gpt_layer_with_transformer_engine_spec(),
                        "pg_collection": pg_collection,
                        "pre_process": True,
                        "post_process": True,
                    },
                )
            },
            "input_projections": [
                ModuleSpec(
                    module=MultimodalProjector,
                    params={
                        "config": proj_cfg,
                        "submodules": MLPSubmodules(
                            linear_fc1=TEColumnParallelLinear,
                            linear_fc2=TERowParallelLinear,
                        ),
                        "projector_type": "mlp",
                        "input_size": vision_config.hidden_size,
                        "tp_group": pg_collection.tp,
                    },
                )
            ],
        },
    )


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def _setup_encoder_dist_opt_groups(encoder_grid, encoder_pg, num_instances):
    """Wire hierarchical process groups for multi-instance encoder optimizer."""
    intra_dp_group, inter_group, intra_dist_opt_group = (
        _create_dist_opt_instance_groups(encoder_grid, num_instances)
    )
    encoder_pg.intra_dp_cp = intra_dp_group
    encoder_pg.intra_expt_dp = intra_dp_group
    encoder_pg.inter_dist_opt = inter_group
    encoder_pg.intra_dist_opt = intra_dist_opt_group
    # The MIMO optimizer creates its own pg_collection from the grid. Store the
    # intra_dist_opt group on the grid so _get_pg_collection_for_optimizer can
    # propagate it without needing access to the DDP wrapper's pg_collection.
    encoder_grid._dist_opt_intra_dist_opt_group = intra_dist_opt_group


def _init_sequence_parallel(llm_pg, config, lp, ep):
    """Initialize RNG tracker, global memory buffer, and TE UB for sequence parallelism."""
    from megatron.core.parallel_state import _set_global_memory_buffer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    llm_tp_rank = dist.get_rank(llm_pg.tp)
    model_parallel_cuda_manual_seed(seed=12345, tp_rank=llm_tp_rank, ep_rank=0, etp_rank=0)
    _set_global_memory_buffer()

    if lp.tp > 1:
        try:
            from transformer_engine.pytorch.module.base import (
                UserBufferQuantizationMode,
                initialize_ub,
            )

            effective_mbs = config.data.micro_batch_size
            if lp.dp > ep.dp:
                effective_mbs = config.data.micro_batch_size * (lp.dp // ep.dp)
            initialize_ub(
                shape=[config.llm_arch.seq_length * effective_mbs, config.llm_arch.hidden_size],
                tp_size=lp.tp,
                quantization_modes=[UserBufferQuantizationMode.NONE],
                ub_cfgs={},
                bootstrap_backend='nccl',
            )
        except Exception as e:
            if dist.get_rank() == 0:
                logger.warning(f"TE UB init failed (tp_comm_overlap disabled): {e}")


def create_mimo_model(config: BenchmarkConfig, pg_manager: ProcessGroupManager):
    """Create a MIMO model with DDP wrapping and process groups.

    Args:
        config: BenchmarkConfig with architecture and parallelism specs.
        pg_manager: ProcessGroupManager for grid and PG lifecycle.

    Returns:
        Tuple of (mimo_model, context_dict) where context_dict contains:
            encoder_grid, llm_grid, encoder_name, encoder_pg, llm_pg,
            and optionally p2p_communicator for PP>1.
    """
    # Clear NVTE env vars that may conflict with TE attention backend detection
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    ep = config.encoder_parallel
    lp = config.llm_parallel

    encoder_grid = pg_manager.create_grid(tp=ep.tp, dp=ep.dp, pp=ep.pp, offset=0)
    llm_grid = pg_manager.create_grid(tp=lp.tp, dp=lp.dp, pp=lp.pp, offset=0)
    pg_manager.create_embedding_groups([encoder_grid, llm_grid])

    torch.manual_seed(12345)

    encoder_pg = pg_manager.get_pg_collection(encoder_grid, is_language_model=False)
    llm_pg = pg_manager.get_pg_collection(llm_grid, is_language_model=True)

    _init_sequence_parallel(llm_pg, config, lp, ep)

    language_model_spec = _get_language_model_spec(config.llm_arch, llm_pg)
    vision_submodule_spec = _get_vision_submodules_spec(
        config.encoder_arch,
        language_hidden_size=config.llm_arch.hidden_size,
        pg_collection=encoder_pg,
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={ENCODER_NAME: vision_submodule_spec},
        special_token_ids={ENCODER_NAME: config.data.image_token_id},
        module_to_grid_map={ENCODER_NAME: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
    )

    mimo_model = MimoModel(mimo_config, tp_group=llm_pg.tp)
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    mimo_model.model_type = ModelType.encoder_or_decoder

    llm_ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True, use_distributed_optimizer=True
    )

    encoder_num_instances = getattr(config, 'encoder_num_dist_opt_instances', 1)
    encoder_ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        bucket_size=200_000_000,
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=encoder_num_instances,
    )

    if encoder_num_instances > 1:
        _setup_encoder_dist_opt_groups(
            encoder_grid, encoder_pg, encoder_num_instances
        )

    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=llm_ddp_config,
            module=mimo_model.language_model,
            pg_collection=llm_pg,
        )

    if ENCODER_NAME in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[ENCODER_NAME]
        if submodule is not None:
            mimo_model.modality_submodules[ENCODER_NAME] = DistributedDataParallel(
                config=submodule.encoders['clip_encoder'].config,
                ddp_config=encoder_ddp_config,
                module=submodule,
                pg_collection=encoder_pg,
            )

    # Attach no_sync / finalize_grads / grad_scale to the config
    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for sub in mimo_model.modality_submodules.values():
                if sub is not None:
                    stack.enter_context(sub.no_sync())
            yield

    def finalize_grads_func(*args, force_all_reduce=False, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=llm_pg,
                force_all_reduce=force_all_reduce,
            )
        for sub in mimo_model.modality_submodules.values():
            if sub is not None:
                finalize_model_grads(
                    [sub],
                    num_tokens=None,
                    pg_collection=encoder_pg,
                    force_all_reduce=force_all_reduce,
                )

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    ctx = {
        'encoder_grid': encoder_grid,
        'llm_grid': llm_grid,
        'encoder_name': ENCODER_NAME,
        'encoder_pg': encoder_pg,
        'llm_pg': llm_pg,
    }

    # Create P2PCommunicator for PP>1
    if config.llm_has_pp:
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

        pp_group = llm_grid.get_pg("pp")
        ctx['p2p_communicator'] = P2PCommunicator(
            pp_group=pp_group, config=mimo_model.config
        )

    return mimo_model, ctx


# ---------------------------------------------------------------------------
# Forward step (PP=1 path only; PP>1 uses colocated_schedule internally)
# ---------------------------------------------------------------------------


def forward_step(data_iterator, model, encoder_grid, llm_grid, encoder_name):
    """Forward step with data slicing for heterogeneous DP.

    Handles three cases:
    - Fan-in (encoder_dp > llm_dp): slice modality_inputs for encoder's smaller batch
    - Fan-out (encoder_dp < llm_dp): slice input_ids/labels/loss_mask for LLM's smaller batch
    - Equal: no slicing

    Args:
        data_iterator: Iterator yielding batch dicts.
        model: MimoModel instance.
        encoder_grid: HyperCommGrid for the encoder.
        llm_grid: HyperCommGrid for the LLM.
        encoder_name: Modality key (e.g. "images").

    Returns:
        Tuple of (output_tensor, loss_func_partial).
    """
    batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}

    if batch.get('input_ids') is None:
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask)

    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if encoder_dp > llm_dp:
        # Fan-in: data loaded with LLM DP (larger batch per rank)
        # Slice modality_inputs for encoder's smaller batch
        scale = encoder_dp // llm_dp
        encoder_dp_idx = encoder_grid.get_pg("dp").rank()
        slot = encoder_dp_idx % scale

        if 'modality_inputs' in batch and batch['modality_inputs'] is not None:
            for mod_name, mod_data in batch['modality_inputs'].items():
                for enc_name, enc_data in mod_data.items():
                    for key, tensor in enc_data.items():
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            # Encoder inputs are [seq, batch, hidden] -- slice batch dim
                            batch_size = tensor.shape[1]
                            slice_size = batch_size // scale
                            start = slot * slice_size
                            enc_data[key] = tensor[
                                :, start : start + slice_size, :
                            ].contiguous()

    elif llm_dp > encoder_dp:
        # Fan-out: slice LLM inputs for LLM's smaller batch
        scale = llm_dp // encoder_dp
        llm_dp_idx = llm_grid.get_pg("dp").rank()
        slot = llm_dp_idx % scale

        batch_size = batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size

        for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][start : start + slice_size].contiguous()

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask)


def loss_func(loss_mask, output_tensor):
    """Compute loss from model output.

    Args:
        loss_mask: Tensor of shape [batch, seq] with 0/1 masking.
        output_tensor: Raw loss tensor from the model, or None.

    Returns:
        Tuple of (scalar_loss, metrics_dict).
    """
    if output_tensor is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}
    loss = output_tensor.float().sum()
    return loss, {'loss_reduced': loss.detach().item()}


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def _compute_effective_mbs(config: BenchmarkConfig, encoder_grid, llm_grid):
    """Compute effective micro-batch size for the schedule.

    For PP=1 the schedule receives data from the iterator; the MBS depends on
    the DP relationship:
    - Fan-in (enc_dp > llm_dp): MBS = config.data.micro_batch_size (LLM DP)
    - Fan-out (llm_dp > enc_dp): MBS = micro_batch_size * (llm_dp // enc_dp)
    - Equal: MBS = config.data.micro_batch_size

    For PP>1 the colocated schedule handles slicing internally, so MBS is
    always config.data.micro_batch_size.
    """
    mbs = config.data.micro_batch_size

    if config.llm_has_pp:
        return mbs

    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if llm_dp > enc_dp:
        return mbs * (llm_dp // enc_dp)

    return mbs


def run_benchmark(
    config: BenchmarkConfig,
    profile_steps: tuple[int, int] | None = None,
    results_dir: str = "./results",
) -> dict:
    """Run benchmark for a single configuration.

    Creates grids, model, optimizer, and data iterator; runs the training loop;
    collects metrics; and cleans up process groups.

    Args:
        config: Fully validated BenchmarkConfig.
        profile_steps: Optional (start, end) tuple of 0-based iteration indices
            to profile with torch.profiler. None means no profiling.
        results_dir: Directory to save profiler output when profiling is enabled.

    Returns:
        Summary dict with median tflops_per_gpu, tokens_per_sec, etc.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger.info(f"Starting benchmark: {config.experiment.name}")

    # 1. Process groups
    pg_manager = ProcessGroupManager()

    # 2. Model + context
    mimo_model, ctx = create_mimo_model(config, pg_manager)

    encoder_grid = ctx['encoder_grid']
    llm_grid = ctx['llm_grid']
    llm_pg = ctx['llm_pg']

    # 3. Data iterator
    effective_mbs = _compute_effective_mbs(config, encoder_grid, llm_grid)

    data_iter = SyntheticVLMIterator(
        encoder_hidden_size=config.encoder_arch.hidden_size,
        image_seq_length=config.encoder_arch.seq_length,
        total_seq_length=config.llm_arch.seq_length,
        micro_batch_size=effective_mbs,
        vocab_size=config.llm_arch.vocab_size,
        image_token_id=config.data.image_token_id,
        encoder_name=ENCODER_NAME,
    )

    # 4. Optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    # 5. Performance monitor
    monitor = PerformanceMonitor(config, world_size)

    if rank == 0:
        logger.info(
            f"  Total FLOPs/iter: {monitor.total_flops:.2e}, "
            f"effective MBS: {effective_mbs}, "
            f"PP: {'yes' if config.llm_has_pp else 'no'}"
        )

    # 6. Training loop
    num_iterations = config.experiment.num_iterations
    optimizer.zero_grad()

    # Set up optional torch.profiler context
    profiler_ctx = None
    if profile_steps is not None:
        from torch.profiler import ProfilerActivity, profile

        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )

    for i in range(num_iterations):
        # Enter profiler for the profiled range
        in_profile_range = (
            profile_steps is not None and profile_steps[0] <= i <= profile_steps[1]
        )
        if in_profile_range and i == profile_steps[0] and profiler_ctx is not None:
            profiler_ctx.__enter__()

        monitor.start_iteration()

        # --- Phase: forward_backward ---
        torch.cuda.synchronize()
        t_fwd_bwd_start = time.time()

        if config.llm_has_pp:
            from megatron.core.models.mimo.colocated_schedule import (
                colocated_forward_backward_with_pp,
            )

            losses = colocated_forward_backward_with_pp(
                mimo_model=mimo_model,
                data_iterator=data_iter,
                num_microbatches=config.data.num_microbatches,
                encoder_grid=encoder_grid,
                llm_grid=llm_grid,
                encoder_name=ENCODER_NAME,
                seq_length=config.llm_arch.seq_length,
                micro_batch_size=effective_mbs,
                p2p_communicator=ctx['p2p_communicator'],
                pg_collection=llm_pg,
            )
        else:
            from megatron.core.pipeline_parallel import schedules

            losses = schedules.forward_backward_no_pipelining(
                forward_step_func=partial(
                    forward_step,
                    encoder_grid=encoder_grid,
                    llm_grid=llm_grid,
                    encoder_name=ENCODER_NAME,
                ),
                data_iterator=data_iter,
                model=[mimo_model],
                num_microbatches=config.data.num_microbatches,
                seq_length=config.llm_arch.seq_length,
                micro_batch_size=effective_mbs,
                forward_only=False,
                pg_collection=llm_pg,
            )

        torch.cuda.synchronize()
        t_fwd_bwd_ms = (time.time() - t_fwd_bwd_start) * 1000.0

        # --- Phase: optimizer_step ---
        torch.cuda.synchronize()
        t_opt_start = time.time()

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t_opt_ms = (time.time() - t_opt_start) * 1000.0

        metrics = monitor.end_iteration(fwd_bwd_ms=t_fwd_bwd_ms, opt_step_ms=t_opt_ms)

        if rank == 0 and (i + 1) % config.experiment.log_interval == 0:
            warmup_tag = " (warmup)" if (i + 1) <= config.experiment.warmup_iterations else ""
            print(
                f"Iter {metrics['iteration']}{warmup_tag}: "
                f"{metrics['tflops_per_gpu']:.1f} TFLOPs/GPU, "
                f"{metrics['tokens_per_sec']:.0f} tok/s, "
                f"{metrics['max_memory_gb']:.1f} GB | "
                f"fwd_bwd: {metrics['fwd_bwd_ms']:.1f}ms, "
                f"opt: {metrics['opt_step_ms']:.1f}ms"
            )

        # Exit profiler after the profiled range
        if in_profile_range and i == profile_steps[1] and profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)
            # Save profiler output
            if rank == 0:
                from benchmarks.mimo_throughput.analyze_profile import (
                    analyze_profile,
                    save_analysis,
                )

                os.makedirs(results_dir, exist_ok=True)

                # Structured analysis report
                desc = (
                    f"{config.experiment.name} | "
                    f"enc TP{config.encoder_parallel.tp}/DP{config.encoder_parallel.dp} | "
                    f"llm TP{config.llm_parallel.tp}/DP{config.llm_parallel.dp}"
                    f"/PP{config.llm_parallel.pp} | "
                    f"mbs={config.data.micro_batch_size} nmb={config.data.num_microbatches}"
                )
                # Get memory waterfall from model (recorded at phase boundaries)
                mem_waterfall = getattr(mimo_model, 'memory_waterfall', None)
                report = analyze_profile(
                    profiler_ctx,
                    config_description=desc,
                    memory_waterfall=mem_waterfall,
                )
                analysis_path = os.path.join(
                    results_dir,
                    f"{config.experiment.name}_analysis_{profile_steps[0]}-{profile_steps[1]}.txt",
                )
                save_analysis(report, analysis_path)
                print(report)

                # Chrome trace JSON (full timeline + memory events)
                trace_path = os.path.join(
                    results_dir,
                    f"{config.experiment.name}_trace_{profile_steps[0]}-{profile_steps[1]}.json",
                )
                profiler_ctx.export_chrome_trace(trace_path)
                print(f"\nPhase report: {analysis_path}")
                print(f"Chrome trace: {trace_path} ({os.path.getsize(trace_path) / 1e6:.0f} MB)")

    # 7. Summary
    summary = monitor.get_summary()

    if rank == 0:
        logger.info(
            f"Benchmark {config.experiment.name} complete: "
            f"{summary['median_tflops_per_gpu']:.1f} TFLOPs/GPU (median), "
            f"{summary['median_tokens_per_sec']:.0f} tok/s (median)"
        )

    # 8. Attach monitor for saving by caller
    summary['_monitor'] = monitor

    # 9. Cleanup
    del optimizer
    del mimo_model
    torch.cuda.empty_cache()
    gc.collect()
    pg_manager.destroy_all()

    return summary
