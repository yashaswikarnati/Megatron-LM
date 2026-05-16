# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer for MIMO models with heterogeneous parallelism."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class ModuleOptimizerInfo:
    """Optimizer info for a single module."""

    optimizer: Optional[MegatronOptimizer]
    grid: Optional[HyperCommGrid]
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
        # Synchronize found_inf across all ranks to prevent deadlock:
        # if encoder ranks detect inf but LLM ranks don't, the early return
        # would skip the all_reduce in get_grad_norm(), causing a hang.
        found_inf_tensor = torch.tensor([found_inf], dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(found_inf_tensor, op=torch.distributed.ReduceOp.MAX)
        found_inf = found_inf_tensor.item() > 0
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
        num_modules = len(self.module_infos)
        zeros_by_module = torch.zeros(num_modules, device="cuda", dtype=torch.float32)

        for i, (name, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer:
                zeros_by_module[i] = float(info.optimizer.count_zeros())

        torch.distributed.all_reduce(zeros_by_module, op=torch.distributed.ReduceOp.MAX)
        return int(zeros_by_module.sum().item())

    @property
    def param_groups(self) -> List[dict]:
        """Combined param groups from all active module optimizers."""
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

        Reassembles param_groups, grad_scaler, and param_state_sharding_type
        that were extracted and saved as ShardedObjects by sharded_state_dict(),
        then delegates to each per-module optimizer's load_state_dict.
        """
        for name, info in self.module_infos.items():
            if not (info.is_active and info.optimizer):
                continue
            module_sd = state_dict.get(name)
            if module_sd is None:
                continue

            for sub_sd, inner_opt in _iter_optimizer_sub_dicts(module_sd, info.optimizer):
                _restore_param_groups(sub_sd, inner_opt, name)
                _restore_param_state_sharding_type(sub_sd)
                _restore_grad_scaler(sub_sd)

            info.optimizer.load_state_dict(module_sd)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        """Build sharded state dict, routing param_groups, grad_scaler, and
        param_state_sharding_type through distributed save as ShardedObjects
        (common.pt is rank-0 only, which misses non-colocated LLM optimizer
        state).
        """
        sharded_state = {}
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer:
                module_sd = info.optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
                replica_id = _get_replica_id(info.pg_collection)

                for idx, (sub_sd, _) in enumerate(
                    _iter_optimizer_sub_dicts(module_sd, info.optimizer)
                ):
                    suffix = f'.{idx}' if idx > 0 else ''
                    _extract_param_groups(sub_sd, name, suffix, replica_id)
                    _extract_param_state_sharding_type(sub_sd, name, suffix, replica_id)
                    _extract_grad_scaler(sub_sd, name, suffix, replica_id)

                # Namespace every internal ShardedBase key with the submodule name
                # so two module optimizers (e.g. 'language' + 'images') don't collide
                # on identical inner keys like 'chained_0.optimizer.distributed.*'.
                add_prefix_for_sharding(module_sd, f'mimo.{name}.')

                sharded_state[name] = module_sd
            else:
                sharded_state[name] = {}
        return sharded_state

    def reload_model_params(self, state_dict=None):
        for opt in self._active_optimizers:
            opt.reload_model_params(state_dict)


def _iter_optimizer_sub_dicts(module_sd, optimizer):
    """Yield (sub_state_dict, inner_optimizer) pairs.

    For a single optimizer, yields (module_sd, optimizer) once.
    For ChainedOptimizer with N>1 inner optimizers, yields
    (module_sd[i], chained_optimizers[i]) for each.
    """
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    if isinstance(optimizer, ChainedOptimizer) and len(optimizer.chained_optimizers) > 1:
        for idx, inner_opt in enumerate(optimizer.chained_optimizers):
            yield module_sd[idx], inner_opt
    else:
        yield module_sd, optimizer


def _extract_param_groups(sub_sd, module_name, suffix, replica_id):
    """Save: extract param_groups from optimizer sub-dict into a ShardedObject."""
    opt_sub = sub_sd.get('optimizer')
    if isinstance(opt_sub, dict) and 'param_groups' in opt_sub:
        pg = deepcopy(opt_sub['param_groups'])
        for group in pg:
            group['params'] = []
        sub_sd[f'_mimo_param_groups{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.param_groups',
            pg,
            (1,),
            (0,),
            replica_id=replica_id,
        )
        del opt_sub['param_groups']
        # Drop the now-empty `optimizer` wrapper. If we left it in place, the
        # empty dict would round-trip through dist_checkpointing's common-state
        # path with no defined behavior on the load side; explicitly removing
        # it pairs with the `setdefault` in `_restore_param_groups` so the load
        # path always rebuilds a clean wrapper. Pattern from
        # https://github.com/NVIDIA/Megatron-LM/pull/4791.
        if not opt_sub:
            del sub_sd['optimizer']


def _extract_grad_scaler(sub_sd, module_name, suffix, replica_id):
    """Save: extract grad_scaler into a ShardedObject."""
    if 'grad_scaler' in sub_sd and sub_sd['grad_scaler'] is not None:
        sub_sd[f'_mimo_grad_scaler{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.grad_scaler',
            sub_sd.pop('grad_scaler'),
            (1,),
            (0,),
            replica_id=replica_id,
        )


def _extract_param_state_sharding_type(sub_sd, module_name, suffix, replica_id):
    """Save: extract param_state_sharding_type into a ShardedObject.

    Plain non-tensor scalars at the per-module level otherwise travel through
    dist_checkpointing's common-state path (rank 0 only), so for non-colocated
    MIMO they are lost on ranks whose module is inactive on rank 0.
    `DistributedOptimizer.load_state_dict` asserts on the missing key, so it
    must round-trip explicitly. Pattern from NVIDIA/Megatron-LM#4791.
    """
    if 'param_state_sharding_type' in sub_sd:
        sub_sd[f'_mimo_param_state_sharding_type{suffix}'] = ShardedObject(
            f'optimizer.mimo.{module_name}{suffix}.param_state_sharding_type',
            sub_sd.pop('param_state_sharding_type'),
            (1,),
            (0,),
            replica_id=replica_id,
        )


def _restore_param_groups(sub_sd, inner_optimizer, module_name):
    """Load: restore param_groups with current param IDs from the inner optimizer."""
    # Find the _mimo_param_groups key (may have a suffix for chained optimizers)
    pg_key = None
    for k in list(sub_sd.keys()):
        if k.startswith('_mimo_param_groups'):
            pg_key = k
            break
    if pg_key is None:
        return

    loaded_pg = sub_sd.pop(pg_key)
    # Get current param IDs from the inner torch optimizer's state_dict
    current_pg = inner_optimizer.optimizer.state_dict()['param_groups']
    if len(loaded_pg) != len(current_pg):
        raise ValueError(
            f"Optimizer '{module_name}': checkpoint has {len(loaded_pg)} param_groups "
            f"but current optimizer has {len(current_pg)}"
        )
    for loaded_g, current_g in zip(loaded_pg, current_pg):
        loaded_g['params'] = current_g['params']
    # `sub_sd['optimizer']` may be absent on load: when the per-module state_dict
    # produced by `DistributedOptimizer.state_dict()` only contains
    # `param_groups` under the 'optimizer' key, `_extract_param_groups` deletes
    # `param_groups` at save time, and the resulting empty dict can be dropped
    # by dist_checkpointing's common-state round-trip on ranks whose active
    # module wasn't on rank 0. `setdefault` lets the restored `param_groups`
    # land in the right place regardless. Pattern from NVIDIA/Megatron-LM#4801.
    sub_sd.setdefault('optimizer', {})['param_groups'] = loaded_pg


def _restore_grad_scaler(sub_sd):
    """Load: restore grad_scaler from ShardedObject key."""
    for k in list(sub_sd.keys()):
        if k.startswith('_mimo_grad_scaler'):
            sub_sd['grad_scaler'] = sub_sd.pop(k)
            break


def _restore_param_state_sharding_type(sub_sd):
    """Load: restore param_state_sharding_type from its ShardedObject key."""
    for k in list(sub_sd.keys()):
        if k.startswith('_mimo_param_state_sharding_type'):
            sub_sd['param_state_sharding_type'] = sub_sd.pop(k)
            break


def _get_replica_id(pg_collection: Optional[ProcessGroupCollection]) -> tuple:
    """Build replica_id tuple for ShardedObject deduplication.

    Returns ``(tp_rank, pp_rank, dp_rank)`` so only ``(0, 0, 0)`` within each
    module's parallelism group is the main replica; all other ranks in the same
    module are non-main replicas of the same object. Order matches
    `make_sharded_object_for_checkpoint` in
    `megatron/core/transformer/utils.py:168-172` and NVIDIA/Megatron-LM#4801.
    """
    assert pg_collection is not None, "pg_collection required for checkpoint replica_id"
    assert (
        hasattr(pg_collection, 'tp') and pg_collection.tp is not None
    ), "pg_collection.tp must be set for checkpoint deduplication"
    assert (
        hasattr(pg_collection, 'pp') and pg_collection.pp is not None
    ), "pg_collection.pp must be set for checkpoint deduplication"
    assert (
        hasattr(pg_collection, 'dp') and pg_collection.dp is not None
    ), "pg_collection.dp must be set for checkpoint deduplication"
    return (pg_collection.tp.rank(), pg_collection.pp.rank(), pg_collection.dp.rank())


def _module_has_trainable_parameters(module) -> bool:
    """Return whether this rank owns any trainable parameters for a module."""
    return module is not None and any(param.requires_grad for param in module.parameters())


def _module_has_any_trainable_parameters(module, pg_collection: ProcessGroupCollection) -> bool:
    """Return whether any rank in the module optimizer group has trainable parameters.

    Without this cross-rank check, `get_mimo_optimizer` would call
    `get_megatron_optimizer` on a module whose params are all frozen on every
    rank (e.g. the language model under stage1 = ``--freeze-vit --freeze-lm``),
    producing a placeholder optimizer that breaks downstream setup. Pattern
    from NVIDIA/Megatron-LM#4790.
    """
    local_has_params = torch.tensor(
        [int(_module_has_trainable_parameters(module))],
        device=torch.cuda.current_device(),
        dtype=torch.int,
    )
    torch.distributed.all_reduce(
        local_has_params, op=torch.distributed.ReduceOp.MAX, group=pg_collection.intra_dist_opt
    )
    return bool(local_has_params.item())


def _get_pg_collection_for_optimizer(grid) -> ProcessGroupCollection:
    """Create ProcessGroupCollection from HyperCommGrid for optimizer use.

    Only fetches process groups required by the optimizer. Assumes all groups
    are pre-created in the grid via grid.create_pg() - does not create any new groups.
    """
    pg = ProcessGroupCollection()
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.intra_dp_cp = pg.dp_cp
    pg.tp = grid.get_pg("tp")
    pg.pp = grid.get_pg("pp")
    pg.mp = grid.get_pg(["tp", "pp"])
    pg.tp_ep_pp = grid.get_pg(["expt_tp", "ep", "pp"])
    pg.expt_dp = grid.get_pg("expt_dp")
    pg.intra_expt_dp = pg.expt_dp
    pg.intra_dist_opt = grid.get_pg(["tp", "cp", "dp", "pp"])
    return pg


def get_mimo_optimizer(mimo_model: "MimoModel", config: OptimizerConfig) -> MimoOptimizer:
    """Create optimizer for MimoModel with heterogeneous parallelism."""
    from megatron.core.optimizer import get_megatron_optimizer

    grid_map = mimo_model.mimo_config.module_to_grid_map
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

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

            # Skip the optimizer build when no rank in this module's
            # intra-dist-opt group has any trainable parameters (e.g. the
            # language model under stage1 = `--freeze-vit --freeze-lm`).
            # Leaving `optimizer = None` lets `MimoOptimizer.is_stub_optimizer`
            # handle the branch correctly, instead of constructing a
            # placeholder DistributedOptimizer that breaks downstream setup.
            module_has_trainable_params = _module_has_any_trainable_parameters(
                module, pg_collection
            )
            if module is not None and module_has_trainable_params:
                assert (
                    not hasattr(module, 'ddp_config')
                    or module.ddp_config is None
                    or module.ddp_config.num_distributed_optimizer_instances == 1
                ), (
                    "MIMO optimizer does not yet support "
                    "num_distributed_optimizer_instances > 1. "
                    f"Module '{module_name}' has "
                    f"{module.ddp_config.num_distributed_optimizer_instances} instances."
                )
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
