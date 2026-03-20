# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Optimizer for MIMO models with heterogeneous parallelism."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedObject
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass
class ModuleOptimizerInfo:
    """Optimizer info for a single module."""

    optimizer: Optional[MegatronOptimizer]
    grid: Any  # HyperCommGrid
    pg_collection: Optional[ProcessGroupCollection]
    is_active: bool


class MimoOptimizer(MegatronOptimizer):
    """
    Optimizer for MimoModel with heterogeneous parallelism.

    Each module gets its own optimizer. Global gradient norm is computed
    across all modules via all_reduce MAX.
    """

    def __init__(self, module_infos: Dict[str, ModuleOptimizerInfo], config: OptimizerConfig):
        self.module_infos = module_infos
        self.config = config
        self._active_optimizers: List[MegatronOptimizer] = [
            info.optimizer
            for info in module_infos.values()
            if info.is_active and info.optimizer is not None
        ]
        self.is_stub_optimizer = len(self._active_optimizers) == 0
        self.optimizer = None  # Base class compat

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        found_inf = False
        for opt in self._active_optimizers:
            found_inf |= opt.prepare_grads()
        return found_inf

    @torch.no_grad()
    def get_grad_norm(self) -> float:
        """Compute global gradient norm across all modules via all_reduce MAX."""
        num_modules = len(self.module_infos)
        norm_sq = torch.zeros(num_modules, device="cuda", dtype=torch.float32)

        for i, (name, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer:
                module_norm = info.optimizer.get_grad_norm() or 0.0
                norm_sq[i] = module_norm**2

        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.MAX)
        return torch.sqrt(norm_sq.sum()).item()

    @torch.no_grad()
    def step(self) -> Tuple[bool, Optional[float], Optional[int]]:
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip with global norm
        for opt in self._active_optimizers:
            if getattr(opt, "is_stub_optimizer", False):
                continue
            params = opt.get_parameters()
            if params and opt.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    params,
                    max_norm=opt.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=opt.config.use_precision_aware_optimizer,
                )

        num_zeros = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        success = True
        for opt in self._active_optimizers:
            success &= opt.step_with_ready_grads()
        return success

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._active_optimizers:
            opt.zero_grad(set_to_none)

    def get_loss_scale(self) -> torch.Tensor:
        if self._active_optimizers:
            return self._active_optimizers[0].get_loss_scale()
        return torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def count_zeros(self) -> int:
        return sum(opt.count_zeros() for opt in self._active_optimizers)

    @property
    def param_groups(self) -> List[dict]:
        """Aggregate param groups from all active optimizers."""
        groups = []
        for opt in self._active_optimizers:
            groups.extend(opt.param_groups)
        return groups

    # Checkpointing

    def state_dict(self):
        return {
            name: info.optimizer.state_dict() if info.is_active and info.optimizer else None
            for name, info in self.module_infos.items()
        }

    def load_state_dict(self, state_dict: Dict):
        """Load per-module optimizer state dicts."""
        for name, info in self.module_infos.items():
            if not (info.is_active and info.optimizer):
                continue
            module_sd = state_dict.get(name)
            if module_sd is None:
                continue
            info.optimizer.load_state_dict(module_sd)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        """Build sharded state dict for all per-module optimizers.

        Wraps non-sharded optimizer metadata (param_groups, step, grad_scaler)
        as ShardedObject so it goes through the sharded file path instead of
        common.pt. Uses the same key/replica_id pattern as DistributedOptimizer's
        dp_zero_gather_scatter path, but keeps fully_sharded_model_space for
        param state tensors (preserving reshardability).

        All ranks produce the same dict key structure (empty {} for inactive
        modules) to avoid common state divergence across ranks.
        """
        sharded_state = {}
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer:
                module_sd = info.optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                # Wrap plain values as ShardedObject keyed by module name.
                # Uses per-module optimizer's DP group for replica_id so only
                # dp_rank=0 writes, others are replicas. Key includes module
                # name for uniqueness across modules on different ranks.
                dp_rank = (
                    info.pg_collection.dp.rank()
                    if info.pg_collection and info.pg_collection.dp
                    else 0
                )
                for key, val in list(module_sd.items()):
                    if not isinstance(val, ShardedBase) and not _has_sharded_base(val):
                        module_sd[key] = ShardedObject(
                            f'optimizer.mimo.{name}.{key}',
                            val,
                            (1,),
                            (0,),
                            replica_id=(0, 0, dp_rank),
                        )
                sharded_state[name] = module_sd
            else:
                sharded_state[name] = {}
        return sharded_state

    def reload_model_params(self, state_dict=None):
        for opt in self._active_optimizers:
            opt.reload_model_params(state_dict)


def _has_sharded_base(obj) -> bool:
    """Check if a nested structure contains any ShardedBase instances."""
    if isinstance(obj, ShardedBase):
        return True
    if isinstance(obj, dict):
        return any(_has_sharded_base(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_sharded_base(v) for v in obj)
    return False


def _get_pg_collection_for_optimizer(grid) -> ProcessGroupCollection:
    """Create ProcessGroupCollection from HyperCommGrid for optimizer use.

    Only fetches process groups required by the optimizer. Assumes all groups
    are pre-created in the grid via grid.create_pg() - does not create any new groups.

    The following groups must be pre-created in the grid before calling this function:
        grid.create_pg(["dp"])
        grid.create_pg(["dp", "cp"])
        grid.create_pg(["tp"])
        grid.create_pg(["tp", "pp"])
        grid.create_pg(["tp", "ep", "pp"])
        grid.create_pg(["dp", "ep"])

    Args:
        grid: HyperCommGrid with pre-created process groups.

    Returns:
        ProcessGroupCollection containing optimizer-required groups:
        - dp: Data parallel group
        - dp_cp: Data parallel with context parallel
        - tp: Tensor parallel group
        - mp: Model parallel group (tp × pp)
        - tp_ep_pp: Expert tensor-model-pipeline group
        - expt_dp: Expert data parallel group
    """
    pg = ProcessGroupCollection()

    # Core groups needed by optimizer
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.tp = grid.get_pg("tp")
    pg.mp = grid.get_pg(["tp", "pp"])

    # Expert groups
    pg.tp_ep_pp = grid.get_pg(["tp", "ep", "pp"])
    pg.expt_dp = grid.get_pg(["dp", "ep"])

    # Distributed optimizer group (same as dp_cp when num_distributed_optimizer_instances == 1)
    # FIXME: Yash - handle multiple optimizer instances
    pg.intra_dist_opt = grid.get_pg(["dp", "cp"])

    return pg


def get_mimo_optimizer(mimo_model: "MimoModel", config: OptimizerConfig) -> MimoOptimizer:
    """Create optimizer for MimoModel with heterogeneous parallelism."""
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
    from megatron.core.optimizer import get_megatron_optimizer

    grid_map = mimo_model.mimo_config.module_to_grid_map
    lang_key = MIMO_LANGUAGE_MODULE_KEY

    module_infos: Dict[str, ModuleOptimizerInfo] = {}

    for module_name, grid in grid_map.items():
        is_active = grid.is_current_rank_in_grid()

        optimizer = None
        pg_collection = _get_pg_collection_for_optimizer(grid)

        if is_active:
            if module_name == lang_key:
                module = mimo_model.language_model
            else:
                module = mimo_model.modality_submodules[module_name]

            if module is not None:
                optimizer = get_megatron_optimizer(
                    config=config,
                    model_chunks=[module],
                    pg_collection=pg_collection,
                    use_gloo_process_groups=False,
                )

        module_infos[module_name] = ModuleOptimizerInfo(
            optimizer=optimizer, grid=grid, pg_collection=pg_collection, is_active=is_active
        )

    return MimoOptimizer(module_infos, config)
