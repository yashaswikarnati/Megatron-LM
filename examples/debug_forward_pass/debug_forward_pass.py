#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Debug a single Megatron-Core GPT forward pass with TP=8 and PP=2.

This script is intentionally small and explicit. It builds a synthetic GPT-style
model, runs one pipeline-parallel forward pass, captures the hidden state after
each transformer layer, gathers those activations to global rank 0, and saves a
single .pt file for offline inspection.
"""

from __future__ import annotations

import argparse
import os
import pdb
import socket
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture layer hidden activations from a single Megatron-Core GPT forward pass."
    )
    parser.add_argument("--tensor-model-parallel-size", "--tp", type=int, default=8)
    parser.add_argument("--pipeline-model-parallel-size", "--pp", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attention-mask-mode",
        choices=("none", "causal", "test-ones"),
        default="none",
        help=(
            "Attention mask passed to GPTModel.forward. The default matches the "
            "pipeline-layout GPT forward tests, where the layer spec supplies causal masking. "
            "'test-ones' matches tests/unit_tests/models/test_gpt_model.py."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("debug_forward_pass_activations.pt"),
        help="Path where global rank 0 writes the gathered activation payload.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into pdb after the forward pass. Other ranks wait at a barrier.",
    )
    parser.add_argument(
        "--interactive-rank",
        type=int,
        default=0,
        help="Global rank that enters pdb when --interactive is set.",
    )
    parser.add_argument(
        "--print-rank-map",
        action="store_true",
        help="Print TP/PP/DP rank mapping at startup.",
    )
    return parser.parse_args()


def require_distributed_environment() -> None:
    missing = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ]
    if missing:
        raise RuntimeError(
            "This example must be launched with torchrun or an equivalent distributed launcher. "
            f"Missing environment variables: {', '.join(missing)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA GPUs and the NCCL backend.")


def initialize_distributed(args: argparse.Namespace) -> torch.device:
    require_distributed_environment()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    model_parallel_world_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )

    if world_size % model_parallel_world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by TP*PP={model_parallel_world_size}."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    return torch.device("cuda", local_rank)


def rank0_print(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def print_rank_map() -> None:
    entries = [
        (
            dist.get_rank(),
            socket.gethostname(),
            int(os.environ["LOCAL_RANK"]),
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )
    ]
    gathered: Optional[List[Any]]
    gathered = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(entries[0], object_gather_list=gathered, dst=0)
    if dist.get_rank() == 0:
        print("Rank map: global_rank host local_rank tp_rank pp_rank dp_rank", flush=True)
        for entry in sorted(gathered or []):
            print("  {} {} {} {} {} {}".format(*entry), flush=True)


def build_transformer_config(args: argparse.Namespace) -> TransformerConfig:
    return TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        sequence_parallel=False,
    )


def build_model(args: argparse.Namespace, config: TransformerConfig) -> GPTModel:
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="learned_absolute",
    )
    model.model_type = ModelType.encoder_or_decoder
    model.eval()
    return model.cuda()


def build_attention_mask(
    args: argparse.Namespace, device: torch.device
) -> Optional[torch.Tensor]:
    if args.attention_mask_mode == "none":
        return None

    if args.attention_mask_mode == "test-ones":
        return torch.ones(
            (args.micro_batch_size, 1, args.seq_length, args.seq_length),
            dtype=torch.bool,
            device=device,
        )

    return torch.triu(
        torch.ones((args.seq_length, args.seq_length), dtype=torch.bool, device=device),
        diagonal=1,
    ).view(1, 1, args.seq_length, args.seq_length)


def build_dummy_batch(
    args: argparse.Namespace, device: torch.device
) -> Dict[str, Optional[torch.Tensor]]:
    token_row = torch.arange(args.seq_length, dtype=torch.long, device=device) % args.vocab_size
    tokens = token_row.unsqueeze(0).repeat((args.micro_batch_size, 1)).contiguous()
    position_ids = (
        torch.arange(args.seq_length, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(args.micro_batch_size, -1)
        .contiguous()
    )

    return {
        "tokens": tokens,
        "position_ids": position_ids,
        "attention_mask": build_attention_mask(args, device),
    }


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def clone_activation(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def register_activation_hooks(
    model: GPTModel,
    activations: MutableMapping[str, torch.Tensor],
    stats: MutableMapping[str, Dict[str, Any]],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    should_capture = (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
    )

    if not should_capture:
        return handles

    for layer in model.decoder.layers:
        layer_number = int(layer.layer_number)
        layer_name = f"layer_{layer_number:03d}"

        def hook(
            _module: torch.nn.Module,
            _inputs: Any,
            output: Any,
            name: str = layer_name,
        ) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states):
                raise TypeError(
                    f"Expected tensor hidden states from {name}, got {type(hidden_states)}"
                )
            captured = clone_activation(hidden_states)
            activations[name] = captured
            stats[name] = tensor_stats(captured)

        handles.append(layer.register_forward_hook(hook))

    return handles


def forward_step_factory(batch: Dict[str, Optional[torch.Tensor]]):
    def collect_output(output_tensor: torch.Tensor, non_loss_data: bool = False) -> Dict[str, Any]:
        del non_loss_data
        return {
            "logits_shape": list(output_tensor.shape),
            "logits_dtype": str(output_tensor.dtype),
            "logits_stats": tensor_stats(output_tensor.detach()),
        }

    def forward_step(_data_iterator: Iterable[Any], model: GPTModel):
        if parallel_state.is_pipeline_first_stage():
            tokens = batch["tokens"]
            position_ids = batch["position_ids"]
        else:
            tokens = None
            position_ids = None

        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=batch["attention_mask"],
            labels=None,
            runtime_gather_output=False,
        )
        return output_tensor, collect_output

    return forward_step


def run_forward_pass(
    args: argparse.Namespace,
    model: GPTModel,
    batch: Dict[str, Optional[torch.Tensor]],
) -> List[Any]:
    forward_backward_func = get_forward_backward_func()
    return forward_backward_func(
        forward_step_func=forward_step_factory(batch),
        data_iterator=iter([None]),
        model=[model],
        num_microbatches=1,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.seq_length,
        forward_only=True,
        collect_non_loss_data=True,
    )


def gather_activation_payloads(
    activations: MutableMapping[str, torch.Tensor],
    stats: MutableMapping[str, Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    local_payload = {
        "rank": dist.get_rank(),
        "hostname": socket.gethostname(),
        "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "dp_rank": parallel_state.get_data_parallel_rank(with_context_parallel=True),
        "activations": OrderedDict(sorted(activations.items())),
        "stats": OrderedDict(sorted(stats.items())),
    }
    gathered: Optional[List[Any]]
    gathered = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(local_payload, object_gather_list=gathered, dst=0)
    return gathered


def gather_forward_data(forward_data: List[Any]) -> List[Any]:
    """Gather the schedule's forward_data_store from the last pipeline stage to rank 0.

    The non-interleaved schedule only appends to forward_data_store on the last pipeline
    stage, so on rank 0 (pp_rank=0) the returned list is empty. We pick the last-stage
    tp_rank=0/dp_rank=0 rank as the source so the captured logits stats correspond to a
    single, well-defined TP shard.
    """
    is_source = (
        parallel_state.is_pipeline_last_stage()
        and parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
    )
    payload = forward_data if is_source else None
    gathered: Optional[List[Any]]
    gathered = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(payload, object_gather_list=gathered, dst=0)
    if dist.get_rank() != 0:
        return []
    for item in gathered or []:
        if item is not None:
            return item
    return []


def merge_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    activations: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    stats: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    sources: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for payload in payloads:
        if not payload["activations"]:
            continue
        for name, tensor in payload["activations"].items():
            if name in activations:
                raise RuntimeError(f"Duplicate captured activation for {name}")
            activations[name] = tensor
            stats[name] = payload["stats"][name]
            sources[name] = {
                "rank": payload["rank"],
                "hostname": payload["hostname"],
                "tp_rank": payload["tp_rank"],
                "pp_rank": payload["pp_rank"],
                "dp_rank": payload["dp_rank"],
            }

    return {
        "activations": OrderedDict(sorted(activations.items())),
        "stats": OrderedDict(sorted(stats.items())),
        "sources": OrderedDict(sorted(sources.items())),
    }


def save_and_print_payload(
    args: argparse.Namespace,
    config: TransformerConfig,
    gathered_payloads: List[Dict[str, Any]],
    forward_data: List[Any],
) -> Dict[str, Any]:
    merged = merge_payloads(gathered_payloads)

    if len(merged["activations"]) != args.num_layers:
        captured = ", ".join(merged["activations"].keys()) or "<none>"
        raise RuntimeError(
            f"Expected {args.num_layers} layer activations, captured "
            f"{len(merged['activations'])}: {captured}"
        )

    output = {
        "metadata": {
            "world_size": dist.get_world_size(),
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
            "data_parallel_size": dist.get_world_size()
            // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size),
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "seq_length": args.seq_length,
            "micro_batch_size": args.micro_batch_size,
            "vocab_size": args.vocab_size,
            "seed": args.seed,
            "attention_mask_mode": args.attention_mask_mode,
            "sequence_parallel": config.sequence_parallel,
            "activation_layout": "[sequence, batch, hidden]",
            "capture_policy": "tp_rank=0 and dp_rank=0 per pipeline stage",
            "forward_data": forward_data,
        },
        "activations": merged["activations"],
        "stats": merged["stats"],
        "sources": merged["sources"],
    }

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.save)

    print(f"Saved activations to {args.save}", flush=True)
    print("Layer activation statistics:", flush=True)
    for name, layer_stats in output["stats"].items():
        shape = "x".join(str(dim) for dim in layer_stats["shape"])
        source = output["sources"][name]
        print(
            f"  {name} pp={source['pp_rank']} rank={source['rank']} "
            f"shape={shape} dtype={layer_stats['dtype']} "
            f"mean={layer_stats['mean']:.6f} std={layer_stats['std']:.6f} "
            f"min={layer_stats['min']:.6f} max={layer_stats['max']:.6f}",
            flush=True,
        )

    return output


def maybe_enter_debugger(
    args: argparse.Namespace,
    local_activations: MutableMapping[str, torch.Tensor],
    local_stats: MutableMapping[str, Dict[str, Any]],
    gathered_payload: Optional[Dict[str, Any]],
) -> None:
    if not args.interactive:
        return

    dist.barrier()
    if dist.get_rank() == args.interactive_rank:
        print(
            f"Rank {dist.get_rank()} entering pdb. Useful locals: "
            "local_activations, local_stats, gathered_payload.",
            flush=True,
        )
        pdb.set_trace()
    dist.barrier()


def main() -> None:
    args = parse_args()
    device = initialize_distributed(args)

    if args.print_rank_map:
        print_rank_map()

    rank0_print(
        "Running debug forward pass with "
        f"TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}, "
        f"layers={args.num_layers}, hidden={args.hidden_size}, heads={args.num_attention_heads}"
    )

    config = build_transformer_config(args)
    model = build_model(args, config)
    batch = build_dummy_batch(args, device)

    local_activations: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    local_stats: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    hooks = register_activation_hooks(model, local_activations, local_stats)

    try:
        with torch.no_grad():
            forward_data = run_forward_pass(args, model, batch)
        torch.cuda.synchronize()

        gathered_payloads = gather_activation_payloads(local_activations, local_stats)
        forward_data = gather_forward_data(forward_data)
        gathered_payload = None
        if dist.get_rank() == 0:
            gathered_payload = save_and_print_payload(
                args, config, gathered_payloads or [], forward_data
            )

        maybe_enter_debugger(args, local_activations, local_stats, gathered_payload)
    finally:
        for handle in hooks:
            handle.remove()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
