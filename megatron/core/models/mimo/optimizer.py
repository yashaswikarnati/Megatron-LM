# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Optimizer for MIMO models with heterogeneous parallelism."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
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
        """Load per-module optimizer state dicts.

        Reassembles param_groups and grad_scaler that were extracted and saved
        as ShardedObjects by sharded_state_dict(), then delegates to each
        per-module optimizer's load_state_dict.
        """
        for name, info in self.module_infos.items():
            if not (info.is_active and info.optimizer):
                continue
            module_sd = state_dict.get(name)
            if module_sd is None:
                continue

            # Reassemble param_groups from the ShardedObject we saved separately.
            # The saved groups have params=[] since LocalNonpersistentObject was
            # stripped at save time. Restore params from the current optimizer's
            # param_groups (these are runtime-local param IDs).
            if '_mimo_param_groups' in module_sd:
                opt_key = _get_optimizer_key(module_sd)
                loaded_pg = module_sd.pop('_mimo_param_groups')
                current_pg = info.optimizer.optimizer.state_dict()['param_groups']
                if len(loaded_pg) != len(current_pg):
                    raise ValueError(
                        f"Optimizer '{name}': checkpoint has {len(loaded_pg)} param_groups "
                        f"but current optimizer has {len(current_pg)}"
                    )
                for loaded_g, current_g in zip(loaded_pg, current_pg):
                    loaded_g['params'] = current_g['params']
                module_sd[opt_key]['param_groups'] = loaded_pg

            if '_mimo_grad_scaler' in module_sd:
                module_sd['grad_scaler'] = module_sd.pop('_mimo_grad_scaler')

            info.optimizer.load_state_dict(module_sd)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        """Build sharded state dict for all per-module optimizers.

        Extracts param_groups and grad_scaler from each per-module optimizer's
        state dict and wraps them as ShardedObjects so they go through the
        distributed save path instead of common.pt.

        This is necessary because common.pt is only written by global rank 0,
        which in non-colocated MiMo is always the encoder's rank 0. Without
        this, the LLM optimizer's param_groups and grad_scaler would never be
        saved (NMFW-33 Issue 3).

        param_groups contain runtime-local param IDs (wrapped as
        LocalNonpersistentObject) that must not be persisted. We strip those
        before wrapping, and restore them from the current optimizer on load.

        All ranks produce the same dict key structure (empty {} for inactive
        modules) to keep the state dict structure consistent across ranks.
        """
        sharded_state = {}
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer:
                module_sd = info.optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                replica_id = _get_replica_id(info.pg_collection)

                # Extract param_groups from the optimizer sub-dict and wrap
                # as ShardedObject. Strip runtime-local param IDs first.
                opt_key = _get_optimizer_key(module_sd)
                opt_sub = module_sd.get(opt_key)
                if isinstance(opt_sub, dict) and 'param_groups' in opt_sub:
                    pg = deepcopy(opt_sub['param_groups'])
                    for group in pg:
                        group['params'] = []
                    module_sd['_mimo_param_groups'] = ShardedObject(
                        f'optimizer.mimo.{name}.param_groups', pg, (1,), (0,), replica_id=replica_id
                    )
                    del opt_sub['param_groups']

                # Extract grad_scaler similarly.
                if 'grad_scaler' in module_sd and module_sd['grad_scaler'] is not None:
                    module_sd['_mimo_grad_scaler'] = ShardedObject(
                        f'optimizer.mimo.{name}.grad_scaler',
                        module_sd.pop('grad_scaler'),
                        (1,),
                        (0,),
                        replica_id=replica_id,
                    )

                sharded_state[name] = module_sd
            else:
                sharded_state[name] = {}
        return sharded_state

    def reload_model_params(self, state_dict=None):
        for opt in self._active_optimizers:
            opt.reload_model_params(state_dict)


def _get_optimizer_key(state_dict: Dict) -> str:
    """Return the optimizer sub-dict key ('optimizer' or legacy 'optimizer_state_dict')."""
    if 'optimizer' in state_dict:
        return 'optimizer'
    return 'optimizer_state_dict'


def _get_replica_id(pg_collection: Optional[ProcessGroupCollection]) -> tuple:
    """Build replica_id tuple for ShardedObject deduplication.

    Includes pp_rank so only one PP stage writes the metadata,
    and dp_rank so only dp_rank=0 writes (others are replicas).
    """
    assert pg_collection is not None, "pg_collection required for checkpoint replica_id"
    assert (
        hasattr(pg_collection, 'pp') and pg_collection.pp is not None
    ), "pg_collection.pp must be set for checkpoint deduplication"
    assert (
        hasattr(pg_collection, 'dp') and pg_collection.dp is not None
    ), "pg_collection.dp must be set for checkpoint deduplication"
    return (0, pg_collection.pp.rank(), pg_collection.dp.rank())


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

    # Core groups needed by optimizer and checkpointing
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.tp = grid.get_pg("tp")
    pg.pp = grid.get_pg("pp")
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
