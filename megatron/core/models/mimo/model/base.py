# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import logging
import warnings
from typing import Any, Dict, Optional

import torch
from torch.profiler import record_function

from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.mimo.comm.colocated_communicator import ColocatedBridgeCommunicator
from megatron.core.models.mimo.config import MimoModelConfig, ModuleMemoryConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout, RankRole
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)


class MimoModel(MegatronModule):
    """Multimodal In/Out Model supporting arbitrary combinations of modalities.

    .. warning::
        **EXPERIMENTAL**: This class is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    .. note::
        This implementation is in development and may undergo API changes.


    This model processes multiple modalities (e.g., vision, audio) alongside text,
    combining their embeddings before passing them through a language model.

    Args:
        mimo_config (MimoModelConfig):
            Configuration for the model, including language model and modality submodules
    """

    def __init__(self, mimo_config: MimoModelConfig, cp_group=None, tp_group=None) -> None:
        """Initialize the multimodal model.

        Example:
            ```python
            # Create a model with default configuration
            model = MimoModel(mimo_config)
            ```
        """
        # Initialize with language model's transformer config for MegatronModule compatibility
        super().__init__(mimo_config.language_model_spec.params['config'])

        warnings.warn(
            "MimoModel is experimental and still under active development. "
            "The API may change without notice in future releases.",
            category=UserWarning,
            stacklevel=2,
        )

        self.mimo_config = mimo_config
        modality_names = list(mimo_config.modality_submodules_spec.keys())
        self.colocated_comms = {}
        self.memory_waterfall: dict = {}  # phase_name -> memory_allocated_bytes
        if mimo_config.module_to_grid_map:
            if self._is_colocated(mimo_config.module_to_grid_map):
                self.role = RankRole.colocated(modality_names + [MIMO_LANGUAGE_MODULE_KEY])
                self._build_colocated_communicators()
            else:
                self.role = RankRole.from_grid_map(mimo_config.module_to_grid_map, modality_names)
        else:
            self.role = RankRole.colocated(modality_names + [MIMO_LANGUAGE_MODULE_KEY])

        # Detect LLM PP>1 for two-phase colocated execution
        self.lm_has_pp = False
        self.lm_is_first_pp_stage = True
        if mimo_config.module_to_grid_map:
            lang_grid = mimo_config.module_to_grid_map.get(MIMO_LANGUAGE_MODULE_KEY)
            if lang_grid and 'pp' in lang_grid.dim_names:
                pp_idx = lang_grid.dim_names.index('pp')
                if lang_grid.shape[pp_idx] > 1:
                    self.lm_has_pp = True
                    pp_group = lang_grid.get_pg('pp')
                    pp_rank = pp_group.rank()
                    pp_size = pp_group.size()
                    self.lm_is_first_pp_stage = pp_rank == 0
                    # Update language module stage info for PP>1
                    from megatron.core.models.mimo.config.role import ModuleStageInfo

                    self.role.modules[MIMO_LANGUAGE_MODULE_KEY] = ModuleStageInfo(
                        is_first_stage=(pp_rank == 0), is_last_stage=(pp_rank == pp_size - 1)
                    )

        # Use special token IDs from the config
        self.special_token_ids = (
            mimo_config.special_token_ids.copy() if mimo_config.special_token_ids else {}
        )

        # Extract language model config for partition adapter
        language_config = mimo_config.language_model_spec.params['config']
        max_seq_len = mimo_config.language_model_spec.params.get('max_sequence_length', 4096)

        self.partition_adapter: Optional[PartitionAdapter] = None
        # Create partition adapter only if parallelism is enabled
        if language_config.context_parallel_size > 1 or language_config.sequence_parallel:
            partition_config = PartitionConfig.from_mp_config(
                mp=language_config,
                max_seq_len=max_seq_len,
                kv_format=mimo_config.kv_format,
                cp_group=cp_group,
                tp_group=tp_group,
            )
            self.partition_adapter = PartitionAdapter(partition_config)

        # Apply memory optimization config before building modules
        self._apply_memory_config()

        # Initialize modality submodules from specifications
        self.modality_submodules = torch.nn.ModuleDict()
        self._initialize_submodules()
        self._initialize_language_model()

        # Collect projection_output flags from any modality config
        self._recompute_projection_output = any(
            mcfg.recompute_projection_output for mcfg in self._modality_memory_configs.values()
        )
        self._offload_projection_output = any(
            mcfg.offload_projection_output for mcfg in self._modality_memory_configs.values()
        )

        # Enable schedule-level offload reset if any MIMO module uses offloading.
        # The schedule checks config.fine_grained_activation_offloading to call
        # off_interface.reset() at the end of each iteration.
        if self._has_mimo_offloading:
            self.config.fine_grained_activation_offloading = True

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Build sharded state dict, bypassing parallel_state global fallbacks.

        Iterates modality_submodules manually (ModuleDict lacks sharded_state_dict)
        and injects dp_cp_group from each module's pg_collection.
        """
        sharded_sd = {}
        for name, module in self.named_children():
            if name == 'modality_submodules':
                # Unwrap DDP, call ModalitySubmodules.sharded_state_dict directly
                # (which injects dp_cp_group from its pg_collection)
                for mod_name, mod in module.items():
                    is_ddp = isinstance(mod, DistributedDataParallel)
                    inner = mod.module if is_ddp else mod
                    child_prefix = f'{prefix}{name}.{mod_name}.'
                    if is_ddp:
                        child_prefix += 'module.'
                    sharded_sd.update(
                        inner.sharded_state_dict(child_prefix, sharded_offsets, metadata)
                    )
            else:
                # Inject dp_cp_group from pg_collection for language_model
                inner = module.module if isinstance(module, DistributedDataParallel) else module
                pg = getattr(inner, 'pg_collection', None)
                mod_metadata = metadata
                if pg is not None:
                    assert (
                        hasattr(pg, 'dp_cp') and pg.dp_cp is not None
                    ), f"pg_collection on '{name}' is missing dp_cp group"
                    mod_metadata = dict(metadata) if metadata else {}
                    mod_metadata['dp_cp_group'] = pg.dp_cp
                sharded_sd.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, mod_metadata
                    )
                )
        return sharded_sd

    def align_embeddings_by_token_positions(
        self,
        modality_embeddings: Dict[str, torch.Tensor],  # [num_embeddings, hidden_dim]
        input_ids: torch.Tensor,  # [bs, seq_len]
        special_token_ids: Dict[str, int],
    ) -> torch.Tensor:
        """Align embeddings from different modalities based on special token positions in input_ids.

        Args:
            modality_embeddings: Dictionary mapping modality names to their embeddings.
                For all modalities: tensor of shape (N, H).
                Shape: (num_tokens_for_modality, hidden_dim)
            input_ids: Input token IDs. Shape: (B, S) or (S,)
                Contains special tokens that mark where each modality's embeddings should go.
                The number of special tokens for each modality should exactly match the number
                of embeddings for that modality.
            special_token_ids: Dictionary mapping modality names to their special token IDs

        Returns:
            Combined embeddings tensor. Shape: (S, B, H)
        """
        # Ensure we have at least one modality
        if not modality_embeddings:
            raise ValueError("No modality embeddings provided. At least one modality is required.")

        logger.debug(f"Merging embeddings for modalities: {list(modality_embeddings.keys())}")

        # Use text embeddings if available, otherwise use any modality
        reference_embeddings = modality_embeddings.get(
            "text", next(iter(modality_embeddings.values()))
        )
        hidden_dim = reference_embeddings.size(-1)
        device = reference_embeddings.device
        dtype = reference_embeddings.dtype

        batch_size, seq_length = input_ids.size()  # input_ids is [B, S]
        logger.debug(
            f"Combined output tensor will have shape: [{seq_length}, {batch_size}, {hidden_dim}]"
        )

        combined_embeddings = torch.zeros(
            (batch_size, seq_length, hidden_dim), dtype=dtype, device=device
        )

        # Process each modality in modality_embeddings
        for modality_name, modality_emb in modality_embeddings.items():
            if modality_name == "text":
                mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
                for token_id in special_token_ids.values():
                    mask &= input_ids != token_id
            elif modality_name in special_token_ids:
                token_id = special_token_ids[modality_name]
                mask = input_ids == token_id
            else:
                raise ValueError(f"No special token ID defined for modality {modality_name}")

            # Validate on GPU without CPU sync (torch.debugging only, no .item())
            num_tokens = mask.sum()
            expected = modality_emb.size(0)
            if torch.is_grad_enabled():
                # Fast GPU-side check: avoid .item() which forces CPU-GPU sync
                torch._assert(
                    num_tokens == expected, f"{modality_name} token count mismatch with embeddings"
                )

            expanded_mask = mask.unsqueeze(-1).expand_as(combined_embeddings)
            combined_embeddings.masked_scatter_(expanded_mask, modality_emb.flatten())

        return combined_embeddings.transpose(0, 1).contiguous()  # [S, B, H]

    def _initialize_submodules(self) -> None:
        """Initialize modality submodules from the ModuleSpec configurations.

        When role is set, only initializes submodules this rank participates in.
        Stage info is passed to from_spec() to conditionally skip projection.
        """
        for modality_name, submodule_spec in self.mimo_config.modality_submodules_spec.items():
            if modality_name not in self.role.modules:
                logger.debug(f"Skipping {modality_name} submodule (not in role)")
                continue

            stage_info = self.role.modules[modality_name]
            is_first_stage = stage_info.is_first_stage
            is_last_stage = stage_info.is_last_stage

            submodule_class = submodule_spec.module
            logger.debug(
                f"Building {modality_name} submodule using {submodule_class.__name__} "
                f"(is_first_stage={is_first_stage}, is_last_stage={is_last_stage})"
            )

            # Pass stage info to from_spec so projections are only built when needed
            submodule = submodule_class.from_spec(
                submodule_spec, is_first_stage=is_first_stage, is_last_stage=is_last_stage
            )

            # Apply MIMO-specific memory flags to submodule
            modality_mcfg = self._modality_memory_configs.get(modality_name)
            if modality_mcfg is not None:
                submodule.recompute_projection = modality_mcfg.recompute_projection
                submodule.offload_projection = modality_mcfg.offload_projection
                submodule.offload_encoder_output = modality_mcfg.offload_encoder_output

            self.modality_submodules[modality_name] = submodule

    def _initialize_language_model(self) -> None:
        """Initialize the language model.

        When role is set, only initializes if this rank participates in language module.
        """
        if not self.role.has_language_module:
            logger.debug("Skipping language model initialization (not in role)")
            self.language_model = None
            return

        logger.debug(
            f"Building language model using {self.mimo_config.language_model_spec.module.__name__}"
        )
        self.language_model = build_module(self.mimo_config.language_model_spec)

    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism.

        This method is required by Megatron's pipeline parallel mechanism.
        It passes the output tensor from the previous stage as input to this stage.

        Args:
            input_tensor: Either:
                - Dict[str, Tensor]: Maps module names to their input tensors (for multi-module PP)
                - Tensor or List[Tensor]: Single tensor for language model (backward compat)

        Returns:
            None
        """
        # The schedule wraps input_tensor in a list (schedules.py:415-416),
        # so unwrap first before checking type.
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]

        # Store dict input for multi-module PP
        if isinstance(input_tensor, dict):
            # P2P recv may return [tensor] (list) for VPP compat — unwrap to tensor
            self.input_tensors = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in input_tensor.items()
            }
            return

        self.input_tensors = input_tensor

        if self.language_model is not None and hasattr(self.language_model, 'set_input_tensor'):
            self.language_model.set_input_tensor(input_tensor)

    def get_text_embeddings(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, special_token_ids: Dict[str, int]
    ) -> torch.Tensor:
        """Get embeddings for text tokens in the input.
        Args:
            input_ids: Input token IDs. Shape: (B, S)
                Contains text tokens and potentially special tokens for other modalities.
            position_ids: Position IDs corresponding to input tokens, used for positional encoding.
                Shape: (B, S)
            special_token_ids: Dictionary mapping modality names to their special token IDs.
                Used to identify non-text tokens in the input_ids.

        Returns:
            torch.Tensor: Embeddings for text tokens.
            Shape: (N, H), where N is the number of text tokens.
        """
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)  # [b, s]
        for special_token_id in special_token_ids.values():
            text_mask &= input_ids != special_token_id

        batch_idx, seq_idx = text_mask.nonzero(as_tuple=True)
        input_ids_text = input_ids[batch_idx, seq_idx].unsqueeze(0)

        position_ids_text = (
            position_ids[batch_idx, seq_idx].unsqueeze(0) if position_ids is not None else None
        )

        text_embeddings = (
            unwrap_model(self.language_model)
            .embedding(input_ids=input_ids_text, position_ids=position_ids_text)
            .squeeze(1)
        )  # Shape: [num_text_tokens, hidden_dim]
        return text_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        packing_kwargs: Optional[dict] = None,
        encoder_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Forward pass through the multimodal model.

        Args:
            input_ids: Input token IDs. Shape: (B, S)
            position_ids: Position IDs. Shape: (B, S)
            attention_mask: Attention mask. Shape: (B, S)
            loss_mask: Loss mask. Shape: (B, S)
            labels: Labels for training. Shape: (B, S)
            modality_inputs: Dictionary mapping modality names to encoder inputs. For example:
                {
                    "images": {
                        "clip_encoder": {"pixel_values": clip_images},
                        "vit_encoder": {"images": vit_images}
                    },
                    "audio": {
                        "whisper_encoder": {"input_features": whisper_features}
                    }
                }
            packing_kwargs: Optional dictionary of kwargs to construct PackedSeqParams
                            if packed_seq_params is not provided. For example:
                                {
                                    "cu_seqlens_q": cu_seqlens,
                                    "cu_seqlens_kv": cu_seqlens,
                                    "cu_seqlens_q_padded": cu_seqlens_padded,
                                    "cu_seqlens_kv_padded": cu_seqlens_padded,
                                    "max_seqlen_q": torch.tensor(
                                        max(seqlens_padded), dtype=torch.int32
                                    ),
                                    "max_seqlen_kv": torch.tensor(
                                        max(seqlens_padded), dtype=torch.int32
                                    ),
                                }

        Returns:
            tuple: (output, loss_mask) where output semantics depend on role:
                - Encoder-only ranks: Dict[str, Tensor] of encoder outputs
                - Language module ranks: language model output (logits or loss)
                - No role (all modules colocated): language model output
        """
        if self.role.mode == ModuleLayout.COLOCATED:
            input_tensors = getattr(self, 'input_tensors', None)

            if self.lm_has_pp and input_tensors is not None:
                # PP>1 non-first stage: hidden states from P2P
                lm_result = self._forward_language_module(
                    input_ids,
                    position_ids,
                    attention_mask,
                    labels,
                    {MIMO_LANGUAGE_MODULE_KEY: input_tensors},
                )
                # Unwrap dict for P2P (schedule uses plain tensors, not dicts)
                if isinstance(lm_result, dict):
                    lm_result = lm_result[MIMO_LANGUAGE_MODULE_KEY]
                return lm_result, loss_mask

            return self._forward_all_modules(
                input_ids,
                position_ids,
                attention_mask,
                loss_mask,
                labels,
                modality_inputs,
                packing_kwargs,
                encoder_embeddings=encoder_embeddings,
            )

        # Get any tensors passed via set_input_tensor
        input_tensors = getattr(self, 'input_tensors', None)

        if self.role.mode == ModuleLayout.NON_COLOCATED:
            if self.role.has_modality_modules:
                return self._forward_encoders(modality_inputs, input_tensors), loss_mask

            if self.role.has_language_module:
                return (
                    self._forward_language_module(
                        input_ids, position_ids, attention_mask, labels, input_tensors
                    ),
                    loss_mask,
                )

            raise RuntimeError(f"Rank has no modules assigned in role: {self.role}")

        raise NotImplementedError(f"Pipeline mode {self.role.mode} is not yet supported")

    def _forward_encoders(
        self,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]],
        input_tensors: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for encoder modules on this rank.

        Args:
            modality_inputs: Raw inputs for each modality (images, audio, etc.)
            input_tensors: Hidden states from previous pipeline stages

        Returns:
            Dict mapping encoder names to their output tensors
        """
        outputs = {}

        for encoder_name in self.role.modality_module_names:
            if encoder_name not in self.modality_submodules:
                continue

            submodule = self.modality_submodules[encoder_name]
            output = submodule.forward(
                encoder_inputs=modality_inputs.get(encoder_name) if modality_inputs else None,
                hidden_states=input_tensors.get(encoder_name) if input_tensors else None,
            )

            if output is not None:
                outputs[encoder_name] = output

        return outputs

    def _forward_language_module(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        input_tensors: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass for language module on this rank.

        Args:
            input_ids: Token IDs
            position_ids: Position IDs
            attention_mask: Attention mask
            labels: Labels for loss computation
            input_tensors: Hidden states or embeddings from previous stage

        Returns:
            Language model output (hidden states, logits, or loss depending on stage)
        """
        lang_name = MIMO_LANGUAGE_MODULE_KEY

        if self.role.is_first_stage(lang_name):
            # First stage: receive encoder embeddings, combine with text, pass to LM
            # Build modality embeddings dict from encoder outputs
            modality_embeddings = {}
            if input_tensors:
                for name, tensor in input_tensors.items():
                    if name != lang_name:
                        modality_embeddings[name] = tensor

            # Get text embeddings
            text_embeddings = self.get_text_embeddings(
                input_ids, position_ids, self.special_token_ids
            )
            modality_embeddings["text"] = text_embeddings

            # Combine all embeddings
            combined_embeddings = self.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=self.special_token_ids,
            )

            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=combined_embeddings,
                labels=labels,
                attention_mask=attention_mask,
            )
        else:
            # Non-first stage: receive hidden states from previous LM stage
            hidden_states = input_tensors.get(lang_name) if input_tensors else None

            # Set input tensor on language model for PP (unwrap DDP to reach GPTModel)
            if hidden_states is not None:
                underlying_lm = unwrap_model(self.language_model)
                if hasattr(underlying_lm, 'set_input_tensor'):
                    underlying_lm.set_input_tensor(hidden_states)

            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=None,
                labels=labels,
                attention_mask=attention_mask,
            )

        # Key output for non-last stages so schedule can route to next LM stage
        if not self.role.is_last_stage(lang_name):
            return {lang_name: lm_output}

        return lm_output

    def _record_mem(self, phase_name: str):
        """Record GPU memory at a phase boundary (no sync, fast CPU read)."""
        self.memory_waterfall[phase_name] = torch.cuda.memory_allocated()

    @staticmethod
    def _is_colocated(module_to_grid_map):
        """Check if all grids span the same ranks (colocated)."""
        grids = list(module_to_grid_map.values())
        first = grids[0]
        return all(g.rank_offset == first.rank_offset and g.size == first.size for g in grids[1:])

    def _build_colocated_communicators(self):
        """Build communicators for each encoder → language edge."""
        grid_map = self.mimo_config.module_to_grid_map
        lang_key = MIMO_LANGUAGE_MODULE_KEY
        lang_grid = grid_map[lang_key]
        for mod_name in self.mimo_config.modality_submodules_spec:
            if mod_name in grid_map and mod_name != lang_key:
                src_grid = grid_map[mod_name]
                if src_grid.size == lang_grid.size:
                    self.colocated_comms[(mod_name, lang_key)] = ColocatedBridgeCommunicator(
                        src_grid=src_grid,
                        dest_grid=lang_grid,
                        src_module_name=mod_name,
                        dest_module_name=lang_key,
                    )

    def _apply_memory_config(self):
        """Apply ModuleMemoryConfig to each module's TransformerConfig.

        Stamps recompute/offload fields from memory_config onto each module's
        TransformerConfig before module construction.  MIMO-specific flags
        (recompute_projection, offload_projection, offload_encoder_output) are
        stored on self for later use by ModalitySubmodules.
        """
        mem_cfg = self.mimo_config.memory_config
        if not mem_cfg:
            self._modality_memory_configs = {}
            self._has_mimo_offloading = False
            return

        self._modality_memory_configs = {}
        self._has_mimo_offloading = False

        for module_name, mcfg in mem_cfg.items():
            if module_name == MIMO_LANGUAGE_MODULE_KEY:
                # Warn if MIMO-boundary flags are set on the LLM entry (they have no effect)
                for flag in (
                    'recompute_projection',
                    'offload_projection',
                    'offload_encoder_output',
                    'recompute_projection_output',
                    'offload_projection_output',
                ):
                    if getattr(mcfg, flag, False):
                        logger.warning(
                            f"memory_config['{MIMO_LANGUAGE_MODULE_KEY}'].{flag}=True "
                            "has no effect on the language module; "
                            "MIMO boundary flags are encoder-only."
                        )
                tc = self.mimo_config.language_model_spec.params['config']
                new_tc = self._rebuild_transformer_config(tc, mcfg)
                if new_tc is not tc:
                    self.mimo_config.language_model_spec.params['config'] = new_tc
                # LLM offload_modules also needs the schedule reset
                if getattr(new_tc, 'fine_grained_activation_offloading', False):
                    self._has_mimo_offloading = True
            else:
                # Encoder module — find and rebuild the TransformerConfig in each encoder spec
                submodule_spec = self.mimo_config.modality_submodules_spec.get(module_name)
                if submodule_spec is None:
                    logger.warning(
                        f"memory_config key '{module_name}' not found in modality_submodules_spec"
                    )
                    continue
                # Warn on conflicting flags
                if mcfg.recompute_projection and mcfg.offload_projection:
                    logger.warning(
                        f"memory_config['{module_name}']: recompute_projection and "
                        "offload_projection are both True; offload_projection will be ignored."
                    )
                encoder_specs = (submodule_spec.submodules or {}).get('encoders', {})
                for enc_spec in (
                    encoder_specs.values() if isinstance(encoder_specs, dict) else [encoder_specs]
                ):
                    enc_params = getattr(enc_spec, 'params', None) or {}
                    enc_tc = enc_params.get('config')
                    if enc_tc is not None:
                        new_tc = self._rebuild_transformer_config(enc_tc, mcfg)
                        if new_tc is not enc_tc:
                            enc_spec.params['config'] = new_tc
                # Store MIMO-specific flags for this modality
                self._modality_memory_configs[module_name] = mcfg
                if (
                    mcfg.offload_projection
                    or mcfg.offload_encoder_output
                    or mcfg.offload_projection_output
                    or mcfg.offload_modules
                ):
                    self._has_mimo_offloading = True

    @staticmethod
    def _rebuild_transformer_config(tc, mcfg: ModuleMemoryConfig):
        """Rebuild a TransformerConfig with memory optimization fields from ModuleMemoryConfig.

        We reconstruct the config rather than mutating it because TransformerConfig.__post_init__
        performs validation and setup that depends on recompute/offload fields.  Direct mutation
        after construction would bypass that logic and leave the config in an inconsistent state.
        """

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

        if not overrides:
            return tc

        return dataclasses.replace(tc, **overrides)

    def _apply_colocated_comms(self, modality_embeddings):
        """Transform encoder embeddings from encoder TP/DP to LLM TP/DP layout."""
        lang_key = MIMO_LANGUAGE_MODULE_KEY
        for modality_name in list(modality_embeddings.keys()):
            comm = self.colocated_comms.get((modality_name, lang_key))
            if comm is not None:
                modality_embeddings[modality_name] = comm.communicate(
                    modality_embeddings[modality_name]
                )
        return modality_embeddings

    def encode_and_communicate(self, modality_inputs):
        """Run encoder forward + colocated TP/DP transform (collective)."""
        modality_embeddings = {}
        for modality_name, submodule in self.modality_submodules.items():
            if (
                modality_inputs
                and modality_name in modality_inputs
                and modality_inputs[modality_name] is not None
            ):
                embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                if embeddings is not None:
                    modality_embeddings[modality_name] = embeddings
        if self.colocated_comms:
            modality_embeddings = self._apply_colocated_comms(modality_embeddings)
        return modality_embeddings

    def _forward_all_modules(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        modality_inputs: Optional[Dict[str, Dict[str, Any]]],
        packing_kwargs: Optional[dict] = None,
        encoder_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Forward pass when all modules are on all ranks (no multi-module PP).

        This is the original behavior, preserved for backward compatibility.
        """
        # Initialize fine-grained offload chunk handler for MIMO-level offloading.
        # Skip if the LLM already has its own offloading — GPTModel.forward() will
        # call init_chunk_handler itself, and double-init corrupts chunk accounting.
        if self._has_mimo_offloading and self.training:
            lm_inner = unwrap_model(self.language_model)
            lm_has_own_offloading = getattr(
                getattr(lm_inner, 'config', None), 'fine_grained_activation_offloading', False
            )
            if not lm_has_own_offloading:
                off_interface.init_chunk_handler(
                    vp_size=1, vp_stage=None, min_offloaded_tensor_size=1024
                )

        # If packing_kwargs is provided, construct PackedSeqParams
        packed_seq_params = None
        if packing_kwargs is not None:
            # Ensure correct dtype for seqlens tensors
            for key in packing_kwargs:
                if 'cu_seqlens' in key and packing_kwargs[key] is not None:
                    packing_kwargs[key] = packing_kwargs[key].to(dtype=torch.int32)
            packed_seq_params = PackedSeqParams(**packing_kwargs)
            packed_seq_params.qkv_format = 'thd'
            logger.debug(f"Packed sequence parameters: {packed_seq_params}")

        # Compute text embeddings FIRST — independent of encoder, enables GPU overlap
        with record_function("mimo::text_embedding"):
            text_embeddings = self.get_text_embeddings(
                input_ids, position_ids, self.special_token_ids
            )
        self._record_mem("mimo::text_embedding")

        if encoder_embeddings is not None:
            modality_embeddings = encoder_embeddings
        else:
            # Process each modality to get embeddings
            modality_embeddings = {}

            with record_function("mimo::encoder_forward"):
                for modality_name, submodule in self.modality_submodules.items():
                    if (
                        modality_inputs
                        and modality_name in modality_inputs
                        and modality_inputs[modality_name] is not None
                    ):
                        embeddings = submodule.forward(
                            encoder_inputs=modality_inputs[modality_name]
                        )
                        if embeddings is not None:
                            modality_embeddings[modality_name] = embeddings
            self._record_mem("mimo::encoder_forward")

            # Apply colocated communication if configured
            if self.colocated_comms:
                with record_function("mimo::bridge_communicate"):
                    modality_embeddings = self._apply_colocated_comms(modality_embeddings)
                self._record_mem("mimo::bridge_communicate")

        modality_embeddings["text"] = text_embeddings

        # 2. Merge embeddings from different modalities
        logger.debug(f"Merging embeddings from {len(modality_embeddings)} modalities")
        with record_function("mimo::align_embeddings"):
            if self._recompute_projection_output and self.training:
                combined_embeddings = tensor_parallel.checkpoint(
                    self.align_embeddings_by_token_positions,
                    False,  # distribute_saved_activations
                    modality_embeddings,
                    input_ids,
                    self.special_token_ids,
                )
            elif self._offload_projection_output and self.training:
                # Use a real tensor from modality_embeddings for the interface
                # so the group-start backward hook stays in the autograd graph.
                ref_tensor = next(iter(modality_embeddings.values()))
                align_off = off_interface(True, ref_tensor, "projection_output")
                with align_off as ref_tensor:
                    combined_embeddings = self.align_embeddings_by_token_positions(
                        modality_embeddings=modality_embeddings,
                        input_ids=input_ids,
                        special_token_ids=self.special_token_ids,
                    )
                combined_embeddings = align_off.group_commit(
                    combined_embeddings, "projection_output"
                )
            else:
                combined_embeddings = self.align_embeddings_by_token_positions(
                    modality_embeddings=modality_embeddings,
                    input_ids=input_ids,
                    special_token_ids=self.special_token_ids,
                )
        self._record_mem("mimo::align_embeddings")
        logger.debug(f"Combined embeddings shape: {combined_embeddings.shape}")

        # 3. If sharding is needed, apply PartitionAdapter.
        # combined_embeddings is [S, B, H]; transpose to [B, S, H] for shard() which expects
        # batch-first layout (required by get_batch_on_this_cp_rank). After CP sharding each
        # rank holds [B, S/cp, H]; transpose back to [S/cp, B, H] for the language model.
        if self.partition_adapter is not None:
            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()  # [B, S, H]
            combined_embeddings, labels, loss_mask, _, packed_seq_params = (
                self.partition_adapter.shard(
                    embeddings=combined_embeddings,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
            )
            # shard() returns embeddings in [B, S/cp, H]; transpose to [S/cp, B, H]
            # which is what the language model expects.
            if combined_embeddings is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

        # 5. Forward pass through language model
        with record_function("mimo::llm_forward"):
            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=combined_embeddings,
                labels=labels,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
            )
        self._record_mem("mimo::llm_forward")

        logger.debug(f"Language model output shape: {lm_output.shape}")

        return lm_output, loss_mask
