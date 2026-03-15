#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MIMO model inference script.

Builds a MimoModel, loads a checkpoint, and runs greedy text generation.
Supports both text-only and vision+text inference.

Usage (via inference.sh):
    # Text-only inference (no dataloader needed):
    bash examples/mimo/scripts/inference/inference.sh <megatron_root> --mode text

    # Vision+text inference (needs energon dataloader):
    bash examples/mimo/scripts/inference/inference.sh <megatron_root> --mode vision_text
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch


def main():
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args, get_tokenizer
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from model_providers.nemotron_moe_vlm import (
        model_provider_nemotron_moe_vlm,
        add_nemotron_moe_vlm_args,
    )

    def extra_args(parser):
        add_nemotron_moe_vlm_args(parser)
        group = parser.add_argument_group("Inference", "Inference options")
        group.add_argument("--mode", type=str, default="text",
                           choices=["text", "vision_text"])
        group.add_argument("--max-out-tokens", type=int, default=30)
        group.add_argument("--prompt", type=str, default=None)
        for arg, kw in [
            ("--dataset-provider", {"type": str, "default": "energon_multimodal"}),
            ("--total-seq-length", {"type": int, "default": 4096}),
            ("--num-workers", {"type": int, "default": 2}),
        ]:
            try:
                parser.add_argument(arg, **kw)
            except argparse.ArgumentError:
                pass
        return parser

    import argparse
    initialize_megatron(extra_args_provider=extra_args)
    args = get_args()
    tokenizer = get_tokenizer()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def pr(*a, **kw):
        if rank == 0:
            print(*a, **kw, flush=True)

    pr(f"=== Mode: {args.mode} | Max tokens: {args.max_out_tokens} ===")

    # ── Build model ──
    use_megatron_load = args.load is not None and getattr(args, 'nemotron_checkpoint', None) is None

    def _provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True, **kwargs):
        return model_provider_nemotron_moe_vlm(
            pre_process=pre_process, post_process=post_process,
            add_encoder=add_encoder, add_decoder=add_decoder,
        )

    model_list = get_model(_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    model = model_list[0]

    if use_megatron_load:
        pr(f">>> Loading checkpoint from {args.load}...")
        from megatron.training.checkpointing import load_checkpoint
        iteration = load_checkpoint(model_list, None, None)
        pr(f">>> Loaded (iteration {iteration})")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    num_gen = args.max_out_tokens
    pad_id = tokenizer.eod

    if args.mode == "text":
        _run_text_only(model, tokenizer, args, pr, num_gen, pad_id)
    else:
        _run_vision_text(model, tokenizer, args, pr, num_gen, pad_id)


def _run_text_only(model, tokenizer, args, pr, num_gen, pad_id):
    """Text-only inference through MimoModel (no images)."""
    prompt = args.prompt or (
        "The theory of general relativity developed by Albert Einstein is the "
        "geometric theory of gravitation providing a unified description of gravity "
        "as a geometric property of space and time. The predictions of general "
        "relativity have been confirmed in all observations and experiments to date. "
        "For example it implies the existence of"
    )
    pr(f"\n>>> Prompt ({len(tokenizer.tokenize(prompt))} tokens): {prompt[:100]}...")

    input_ids = torch.tensor([tokenizer.tokenize(prompt)], dtype=torch.long, device="cuda")
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    # Single forward — logits
    with torch.no_grad():
        output, _ = model(
            input_ids=input_ids, position_ids=position_ids,
            attention_mask=None, labels=None, loss_mask=None,
        )
    _print_top10(output, tokenizer, pr)

    generated = _greedy_generate(model, input_ids, position_ids, tokenizer, num_gen, pad_id, pr)
    text = tokenizer.detokenize(generated)
    pr(f"\n>>> Generated ({len(generated)} tokens, {len(set(generated))} unique):")
    pr(f">>> {prompt}{text}")


def _run_vision_text(model, tokenizer, args, pr, num_gen, pad_id):
    """Vision+text inference through MimoModel with energon dataloader."""
    from train import get_batch
    from data.energon_multimodal_provider import train_valid_test_dataloaders_provider

    train_valid_test_dataloaders_provider.is_distributed = True
    train_dl, _, _ = train_valid_test_dataloaders_provider(train_val_test_num_samples=[1, 0, 0])
    data_iter = iter(train_dl) if train_dl is not None else None

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def _to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, dict):
            return {k: _to_cuda(v) for k, v in obj.items()}
        return obj

    # Find a batch with images and loss tokens.
    pr("\n>>> Fetching batch with images...")
    batch = None
    for attempt in range(20):
        raw = get_batch(data_iter)
        if raw is None:
            break
        raw = _to_cuda(raw)
        n_img = (raw["input_ids"][0] == image_token_id).sum().item()
        n_loss = int(raw["loss_mask"].sum().item())
        has_images = raw.get("modality_inputs") is not None
        pr(f"  Batch {attempt}: img_tokens={n_img} loss_tokens={n_loss} has_images={has_images}")
        if n_loss > 0 and has_images:
            batch = raw
            break
    if batch is None:
        pr(">>> ERROR: No valid batch found with images and loss tokens")
        return

    # Forward with labels (loss)
    with torch.no_grad():
        output, _ = model(**batch)
    lm = batch["loss_mask"].contiguous().view(-1).float()
    n_loss = int(lm.sum().item())
    if n_loss > 0:
        avg_loss = torch.sum(output.float().view(-1) * lm) / lm.sum()
        pr(f">>> Loss: {avg_loss.item():.4f} ({n_loss} tokens)")

    # Forward without labels (logits)
    gen_batch = {k: v for k, v in batch.items() if k != "labels"}
    gen_batch["labels"] = None
    with torch.no_grad():
        logits, _ = model(**gen_batch)
    _print_top10(logits, tokenizer, pr)

    # Generate
    gen_ids = batch["input_ids"].clone()
    gen_pos = batch.get("position_ids")
    if gen_pos is None:
        gen_pos = torch.arange(gen_ids.shape[1], device=gen_ids.device).unsqueeze(0)
    mod_inputs = batch.get("modality_inputs")

    generated = _greedy_generate(
        model, gen_ids, gen_pos, tokenizer, num_gen, pad_id, pr,
        modality_inputs=mod_inputs,
    )
    text = tokenizer.detokenize(generated)
    pr(f"\n>>> Generated ({len(generated)} tokens, {len(set(generated))} unique):")
    pr(f">>> {text[:300]}")


def _greedy_generate(model, input_ids, position_ids, tokenizer, num_gen, pad_id, pr,
                     modality_inputs=None):
    """Greedy autoregressive generation. Returns list of generated token IDs."""
    pr(f"\n>>> Generating {num_gen} tokens...")
    gen_ids = input_ids.clone()
    gen_pos = position_ids.clone()
    generated = []

    for step in range(num_gen):
        step_batch = {
            "input_ids": gen_ids,
            "position_ids": gen_pos,
            "attention_mask": None,
            "labels": None,
            "loss_mask": None,
        }
        # Pass modality_inputs only on first step (embeddings cached after).
        if step == 0 and modality_inputs is not None:
            step_batch["modality_inputs"] = modality_inputs

        with torch.no_grad():
            step_out, _ = model(**step_batch)

        next_tok = step_out[0, -1, :].argmax(dim=-1).item()
        generated.append(next_tok)
        if next_tok == tokenizer.eod:
            pr(f">>> EOS at step {step + 1}")
            break

        new_len = gen_ids.shape[1] + 1
        new_ids = torch.full((1, new_len), pad_id, dtype=torch.long, device=gen_ids.device)
        new_ids[0, :gen_ids.shape[1]] = gen_ids[0]
        new_ids[0, -1] = next_tok
        gen_ids = new_ids
        gen_pos = torch.arange(new_len, device=gen_ids.device).unsqueeze(0)

    return generated


def _print_top10(logits, tokenizer, pr):
    """Print top-10 next token predictions from last position."""
    last_logits = logits[0, -1, :]
    top_vals, top_ids = torch.topk(last_logits, 10)
    pr(">>> Top-10 next tokens:")
    for i, (tid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist())):
        pr(f"  {i+1:2d}. {repr(tokenizer.detokenize([tid])):25s} (id={tid:6d}, logit={val:.2f})")


if __name__ == "__main__":
    main()
