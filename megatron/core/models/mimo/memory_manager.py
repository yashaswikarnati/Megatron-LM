# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MimoMemoryManager: recompute and offload for MIMO combined_embeddings.

Handles two concerns:
  1. Config stamping: applies recompute/offload fields from ModuleMemoryConfig
     onto each module's TransformerConfig before construction.
  2. Combined embeddings optimization: either recompute (checkpoint) or offload
     (PipelineOffloadManager) the align_embeddings_by_token_positions call.

ModalitySubmodules is never touched. All MIMO-boundary memory logic lives
in _forward_all_modules (one call site) and the colocated schedule.
"""

import dataclasses
import logging
from contextlib import contextmanager

from megatron.core import tensor_parallel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)

logger = logging.getLogger(__name__)


class MimoMemoryManager:
    """Manages memory optimization for MIMO models.

    Null-object pattern: when no memory_config is provided, all methods are no-ops.
    """

    def __init__(self, mimo_config):
        self._recompute_combined = False
        self._offload_combined = False
        self._needs_offload = False

        memory_config = mimo_config.memory_config
        if memory_config:
            self._apply(memory_config, mimo_config)

    # ------------------------------------------------------------------
    # Config stamping
    # ------------------------------------------------------------------

    def _apply(self, memory_config, mimo_config):
        for module_name, mcfg in memory_config.items():
            # Stamp TransformerConfig fields onto the module spec
            overrides = _tc_overrides(mcfg)
            if module_name == MIMO_LANGUAGE_MODULE_KEY:
                if overrides:
                    tc = mimo_config.language_model_spec.params['config']
                    mimo_config.language_model_spec.params['config'] = dataclasses.replace(
                        tc, **overrides
                    )
            else:
                spec = mimo_config.modality_submodules_spec.get(module_name)
                if spec is None:
                    logger.warning(
                        f"memory_config['{module_name}'] not in modality_submodules_spec"
                    )
                    continue
                if overrides:
                    for enc_spec in _iter_encoder_specs(spec):
                        tc = (getattr(enc_spec, 'params', None) or {}).get('config')
                        if tc is not None:
                            enc_spec.params['config'] = dataclasses.replace(tc, **overrides)

            # MIMO boundary flags (from any module entry — typically encoder)
            if mcfg.recompute_combined_embeddings:
                self._recompute_combined = True
            if mcfg.offload_combined_embeddings:
                self._offload_combined = True
                self._needs_offload = True

            # offload_modules on any module also requires lifecycle
            if mcfg.offload_modules:
                self._needs_offload = True

    # ------------------------------------------------------------------
    # Combined embeddings: recompute or offload
    # ------------------------------------------------------------------

    def forward_combined_embeddings(
        self, align_fn, modality_embeddings, input_ids, special_token_ids
    ):
        """Run align_embeddings_by_token_positions with configured optimization.

        Plain:    returns align_fn(modality_embeddings, input_ids, special_token_ids)
        Recompute: checkpoints align_fn — frees combined_embeddings intermediate
        Offload:   wraps with FineGrainedActivationOffloadingInterface — frees both
                   combined_embeddings AND projection_output via CPU offload
        """
        if not self._recompute_combined and not self._offload_combined:
            return align_fn(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )

        if self._recompute_combined:
            return _checkpoint_align(align_fn, modality_embeddings, input_ids, special_token_ids)

        # Offload path: wrap with FineGrainedActivationOffloadingInterface
        ref_tensor = next(iter(modality_embeddings.values()))
        iface = off_interface(True, ref_tensor, "combined_embeddings")
        with iface as ref_tensor:
            result = align_fn(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
            )
        return iface.group_commit(
            result, "combined_embeddings", forced_released_tensors=[ref_tensor]
        )

    # ------------------------------------------------------------------
    # Offload lifecycle
    # ------------------------------------------------------------------

    @property
    def needs_offload_lifecycle(self):
        """Whether PipelineOffloadManager is needed (for schedule reset)."""
        return self._needs_offload

    def init_offload(self):
        """Initialize PipelineOffloadManager chunk handler for this forward pass.

        Creates a chunk for MIMO-boundary offload groups. If GPTModel also has
        offloading, it creates its own chunk — the singleton handles multiple
        chunks per forward pass (VPP mechanism). After warmup, this is a no-op.
        """
        if self._needs_offload:
            off_interface.init_chunk_handler(
                vp_size=1, vp_stage=None, min_offloaded_tensor_size=1024 * 1024
            )

    @contextmanager
    def deferred_offload_reset(self, config):
        """Suppress the schedule's off_interface.reset() during the body.

        Used by the colocated 3-phase schedule: Phase 3's encoder backward
        needs offloaded tensors from Phase 1 to remain accessible. The reset
        is deferred to after Phase 3 completes.
        """
        if not self._needs_offload:
            yield
            return

        saved = getattr(config, 'fine_grained_activation_offloading', False)
        config.fine_grained_activation_offloading = False
        try:
            yield
        finally:
            config.fine_grained_activation_offloading = saved
            off_interface.reset()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _checkpoint_align(align_fn, modality_embeddings, input_ids, special_token_ids):
    """Checkpoint align_embeddings — saves inputs as Tensors, frees intermediates."""
    keys = list(modality_embeddings.keys())
    tensors = [modality_embeddings[k] for k in keys]

    def _run(*all_tensors):
        embs = dict(zip(keys, all_tensors[:-1]))
        return align_fn(
            modality_embeddings=embs, input_ids=all_tensors[-1], special_token_ids=special_token_ids
        )

    return tensor_parallel.checkpoint(_run, False, *tensors, input_ids)


def _tc_overrides(mcfg):
    """TransformerConfig field overrides from a ModuleMemoryConfig."""
    overrides = {}
    if mcfg.recompute_granularity is not None:
        overrides['recompute_granularity'] = mcfg.recompute_granularity
        overrides['recompute_method'] = mcfg.recompute_method
        overrides['recompute_num_layers'] = mcfg.recompute_num_layers
        if mcfg.recompute_modules is not None:
            overrides['recompute_modules'] = mcfg.recompute_modules
    if mcfg.offload_modules is not None:
        overrides['fine_grained_activation_offloading'] = True
        overrides['offload_modules'] = mcfg.offload_modules
    return overrides


def _iter_encoder_specs(submodule_spec):
    encoder_specs = (submodule_spec.submodules or {}).get('encoders', {})
    return encoder_specs.values() if isinstance(encoder_specs, dict) else [encoder_specs]
