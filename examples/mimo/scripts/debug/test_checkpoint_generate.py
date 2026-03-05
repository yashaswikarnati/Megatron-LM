# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint sanity check: load model, get one data sample, compute loss,
then autoregressively generate tokens and print the decoded text.

This reuses the full Megatron initialization (parallelism, model build,
checkpoint loading) so the test exercises the exact same code path as training.

Usage: launched via test_checkpoint_generate.sh (see that file for CLI args).
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model

# Reuse the existing model/data providers from train.py.
from train import (
    add_mimo_args,
    get_batch,
    model_provider,
    train_valid_test_datasets_provider,
)


def add_generate_args(parser):
    """Add generation-specific args on top of the MIMO args."""
    add_mimo_args(parser)
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate autoregressively",
    )
    return parser


def _is_rank0():
    return parallel_state.get_tensor_model_parallel_rank() == 0 and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


def _print_rank0(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


@torch.no_grad()
def generate_greedy(model, batch, num_tokens, tokenizer, pad_token_id, image_token_id):
    """Autoregressive greedy generation from a data batch.

    Works with the full MIMO model: processes images via vision encoder on the
    first forward pass, then extends only the text tokens for subsequent steps.

    For simplicity this does full-sequence forward passes (no KV cache).
    """
    model.eval()
    args = get_args()

    input_ids = batch["input_ids"]  # [B, S]
    labels = batch["labels"]        # [B, S]
    B, S = input_ids.shape

    # -- Step 1: Diagnose the batch (labels, loss_mask) --
    _print_rank0("\n" + "=" * 70)
    _print_rank0("STEP 1: Batch diagnostics")
    _print_rank0("=" * 70)

    lm = batch["loss_mask"].contiguous().view(-1).float()
    total_loss_tokens = lm.sum().item()
    _print_rank0(f"  input_ids shape:  {input_ids.shape}")
    _print_rank0(f"  labels shape:     {labels.shape}")
    _print_rank0(f"  loss_mask sum:    {total_loss_tokens} / {lm.numel()} "
                 f"({100*total_loss_tokens/max(lm.numel(),1):.1f}% valid)")

    # Check labels distribution.
    flat_labels = labels.view(-1)
    n_ignore = (flat_labels == -100).sum().item()
    n_pad = (flat_labels == pad_token_id).sum().item()
    n_image = (flat_labels == image_token_id).sum().item()
    n_valid = flat_labels.numel() - n_ignore
    _print_rank0(f"  labels: {flat_labels.numel()} total, {n_ignore} ignore(-100), "
                 f"{n_valid} valid, {n_pad} pad, {n_image} image_token")

    # Check if labels are shifted relative to input_ids (next-token prediction).
    # Compare labels[i] vs input_ids[i+1] for non-ignore positions.
    ids_flat = input_ids[0, :S-1]      # [S-1]
    lab_flat = labels[0, :S-1]         # [S-1]
    ids_shifted = input_ids[0, 1:S]    # [S-1] = input_ids shifted left by 1
    valid_mask = lab_flat != -100
    if valid_mask.any():
        # labels[i] == input_ids[i] means labels are NOT shifted (same-token)
        same_token_match = (lab_flat[valid_mask] == ids_flat[valid_mask]).float().mean().item()
        # labels[i] == input_ids[i+1] means labels ARE shifted (next-token)
        next_token_match = (lab_flat[valid_mask] == ids_shifted[valid_mask]).float().mean().item()
        _print_rank0(f"  Label shift check (first sample, valid positions):")
        _print_rank0(f"    labels[i] == input_ids[i]   (same-token): {same_token_match:.3f}")
        _print_rank0(f"    labels[i] == input_ids[i+1] (next-token): {next_token_match:.3f}")
        if same_token_match > 0.8:
            _print_rank0("    >>> Labels appear UNSHIFTED — loss will be meaningless!")
            _print_rank0("    >>> Labels should be next-token targets: labels[i] = input_ids[i+1]")
        elif next_token_match > 0.8:
            _print_rank0("    >>> Labels appear correctly shifted (next-token prediction).")
        else:
            _print_rank0("    >>> Labels don't match either pattern — check data provider.")

        # Show a few examples.
        valid_indices = valid_mask.nonzero(as_tuple=True)[0][:10]
        _print_rank0(f"    Sample positions (first 10 valid):")
        for idx in valid_indices:
            i = idx.item()
            _print_rank0(f"      pos {i}: input_ids={ids_flat[i].item()}, "
                         f"labels={lab_flat[i].item()}, input_ids[i+1]={ids_shifted[i].item()}")
    else:
        _print_rank0("  WARNING: No valid label positions found! loss_mask is all zeros.")
        _print_rank0("  This means avg_loss will be 0 by definition (0/clamp(0,1)=0).")

    # -- Step 2: compute loss on the original sample --
    _print_rank0("\n" + "=" * 70)
    _print_rank0("STEP 2: Forward pass with labels (loss check)")
    _print_rank0("=" * 70)

    output_tensor, model_loss_mask = model(**batch)
    losses = output_tensor.float()
    _print_rank0(f"  output_tensor shape: {output_tensor.shape}")
    _print_rank0(f"  output_tensor stats: min={losses.min().item():.4f}, "
                 f"max={losses.max().item():.4f}, mean={losses.mean().item():.4f}")

    total_loss = torch.sum(losses.view(-1) * lm)
    total_tokens = lm.sum()
    avg_loss = total_loss / total_tokens.clamp(min=1)

    _print_rank0(f"  Average loss:  {avg_loss.item():.4f}")
    _print_rank0(f"  Loss tokens:   {int(total_tokens.item())}")
    _print_rank0(f"  Perplexity:    {torch.exp(avg_loss).item():.2f}")

    ln_vocab = torch.log(torch.tensor(float(args.padded_vocab_size)))
    if total_tokens.item() == 0:
        _print_rank0("  WARNING: loss_mask has NO valid tokens — loss is trivially 0!")
        _print_rank0("  This likely means all labels are -100 (ignore_index).")
    elif avg_loss > ln_vocab:
        _print_rank0(
            f"  WARNING: loss ({avg_loss.item():.2f}) > ln(vocab_size) "
            f"({ln_vocab.item():.2f}) — model may not have loaded correctly!"
        )

    # -- Step 3: autoregressive generation --
    _print_rank0("\n" + "=" * 70)
    _print_rank0("STEP 3: Autoregressive greedy generation")
    _print_rank0("=" * 70)

    # Find where actual tokens end (before padding).
    non_pad_mask = input_ids != pad_token_id
    seq_lengths = non_pad_mask.sum(dim=1)  # [B]
    prompt_len = seq_lengths[0].item()

    # Decode the prompt for context.
    prompt_token_ids = input_ids[0, :prompt_len].tolist()
    text_token_ids = [t for t in prompt_token_ids if t != image_token_id]
    num_image_tokens = prompt_len - len(text_token_ids)

    _print_rank0(f"  Prompt length: {prompt_len} tokens "
                 f"({len(text_token_ids)} text + {num_image_tokens} image)")

    prompt_text = ""
    if text_token_ids:
        prompt_text = tokenizer.detokenize(text_token_ids)
        if len(prompt_text) > 500:
            _print_rank0(f"  Prompt (first 250 chars): {prompt_text[:250]}")
            _print_rank0(f"  Prompt (last 250 chars):  ...{prompt_text[-250:]}")
        else:
            _print_rank0(f"  Prompt: {prompt_text}")

    # -- Generate tokens one at a time --
    gen_input_ids = input_ids[0:1, :prompt_len].clone()  # [1, prompt_len]
    gen_position_ids = torch.arange(prompt_len, device=gen_input_ids.device).unsqueeze(0)

    gen_modality_inputs = None
    if "modality_inputs" in batch and batch["modality_inputs"] is not None:
        gen_modality_inputs = batch["modality_inputs"]

    generated_ids = []
    eos_token_id = tokenizer.eod

    for step in range(num_tokens):
        gen_batch = {
            "input_ids": gen_input_ids,
            "position_ids": gen_position_ids,
            "attention_mask": None,
            "labels": None,  # No labels → returns logits
            "loss_mask": None,
        }
        if gen_modality_inputs is not None:
            gen_batch["modality_inputs"] = gen_modality_inputs

        logits, _ = model(**gen_batch)

        next_token_logits = logits[0, -1, :]  # [V]
        next_token = next_token_logits.argmax(dim=-1).item()

        generated_ids.append(next_token)

        if next_token == eos_token_id:
            _print_rank0(f"  EOS reached at step {step + 1}")
            break

        # Extend the sequence (no KV cache, full recompute each step).
        new_len = gen_input_ids.shape[1] + 1
        new_input_ids = torch.full(
            (1, new_len), pad_token_id, dtype=gen_input_ids.dtype, device=gen_input_ids.device
        )
        new_input_ids[0, :gen_input_ids.shape[1]] = gen_input_ids[0]
        new_input_ids[0, -1] = next_token
        gen_input_ids = new_input_ids
        gen_position_ids = torch.arange(new_len, device=gen_input_ids.device).unsqueeze(0)

        if (step + 1) % 10 == 0:
            _print_rank0(f"  Generated {step + 1}/{num_tokens} tokens...")

    # -- Decode and display generated text --
    _print_rank0(f"\n  Generated {len(generated_ids)} tokens.")

    generated_text = tokenizer.detokenize(generated_ids)
    _print_rank0(f"\n  --- Generated text ---")
    _print_rank0(f"  {generated_text}")
    _print_rank0(f"  --- End generated text ---")

    _print_rank0(f"\n  Token IDs: {generated_ids[:50]}{'...' if len(generated_ids) > 50 else ''}")

    # -- Quality checks --
    _print_rank0("\n" + "=" * 70)
    _print_rank0("SUMMARY")
    _print_rank0("=" * 70)
    _print_rank0(f"  Loss:           {avg_loss.item():.4f}")
    _print_rank0(f"  Perplexity:     {torch.exp(avg_loss).item():.2f}")
    _print_rank0(f"  Generated:      {len(generated_ids)} tokens")

    unique_tokens = len(set(generated_ids))
    _print_rank0(f"  Unique tokens:  {unique_tokens}/{len(generated_ids)}")
    if unique_tokens <= 2 and len(generated_ids) > 10:
        _print_rank0("  WARNING: Generation is degenerate (repetitive single token)")
    if total_tokens.item() == 0:
        _print_rank0("  WARNING: No valid loss tokens — cannot assess checkpoint quality via loss")
    elif avg_loss.item() > ln_vocab.item():
        _print_rank0("  WARNING: Loss is worse than random — checkpoint may not have loaded")
    elif avg_loss.item() > 5.0:
        _print_rank0("  WARNING: Loss is high — checkpoint may be partially loaded")
    else:
        _print_rank0("  OK: Loss looks reasonable for a pre-trained model")

    _print_rank0("=" * 70)

    # -- Dump diagnostics to JSON --
    if _is_rank0():
        _dump_json(
            args=args,
            batch=batch,
            input_ids=input_ids,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            text_token_ids=text_token_ids,
            generated_ids=generated_ids,
            generated_text=generated_text,
            avg_loss=avg_loss.item(),
            total_loss_tokens=int(total_tokens.item()),
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
        )


def _dump_json(
    args, batch, input_ids, prompt_text, prompt_token_ids, text_token_ids,
    generated_ids, generated_text, avg_loss, total_loss_tokens,
    image_token_id, pad_token_id,
):
    """Dump all diagnostics to a JSON file in logs/generations/."""
    out_dir = os.path.join(os.getcwd(), "logs", "generations")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"generation_{timestamp}.json")

    B, S = input_ids.shape
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]

    # Save actual image tiles as PNGs.
    image_info = {"tile_paths": [], "num_tiles": 0}
    tiles_dir = os.path.join(out_dir, f"tiles_{timestamp}")
    if "modality_inputs" in batch and batch["modality_inputs"] is not None:
        for mod_name, mod_inputs in batch["modality_inputs"].items():
            for enc_name, enc_inputs in mod_inputs.items():
                for inp_name, inp_val in enc_inputs.items():
                    if isinstance(inp_val, torch.Tensor) and inp_val.ndim == 4:
                        # inp_val: (num_tiles, C, H, W) — save each tile as PNG.
                        os.makedirs(tiles_dir, exist_ok=True)
                        tiles = inp_val.cpu().float()
                        image_info["num_tiles"] = tiles.shape[0]
                        image_info["tile_shape"] = list(tiles.shape[1:])
                        image_info["dtype"] = str(inp_val.dtype)
                        for t_idx in range(tiles.shape[0]):
                            tile = tiles[t_idx]  # (C, H, W)
                            # Normalize to [0, 255] for saving.
                            tile = tile - tile.min()
                            denom = tile.max()
                            if denom > 0:
                                tile = tile / denom
                            tile = (tile * 255).clamp(0, 255).byte()
                            # (C, H, W) → (H, W, C) for PIL.
                            tile_np = tile.permute(1, 2, 0).numpy()
                            tile_path = os.path.join(tiles_dir, f"tile_{t_idx:03d}.png")
                            try:
                                from PIL import Image
                                img = Image.fromarray(tile_np)
                                img.save(tile_path)
                            except ImportError:
                                # Fallback: save raw tensor as .pt file.
                                tile_path = os.path.join(tiles_dir, f"tile_{t_idx:03d}.pt")
                                torch.save(tiles[t_idx], tile_path)
                            image_info["tile_paths"].append(tile_path)
        if image_info["tile_paths"]:
            print(f"  Saved {len(image_info['tile_paths'])} image tiles to {tiles_dir}/")

    result = {
        "timestamp": timestamp,
        "checkpoint": getattr(args, "nemotron_checkpoint", None),
        "model_config": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "num_experts": getattr(args, "num_experts", None),
            "tp": args.tensor_model_parallel_size,
            "ep": args.expert_model_parallel_size,
        },
        "batch_info": {
            "input_ids_shape": list(input_ids.shape),
            "labels_shape": list(labels.shape),
            "loss_mask_sum": int(loss_mask.sum().item()),
            "loss_mask_total": int(loss_mask.numel()),
            "n_image_tokens": int((input_ids[0] == image_token_id).sum().item()),
            "n_pad_tokens": int((input_ids[0] == pad_token_id).sum().item()),
            "n_text_tokens": int(((input_ids[0] != image_token_id) & (input_ids[0] != pad_token_id)).sum().item()),
        },
        "loss": {
            "avg_loss": avg_loss,
            "perplexity": float(torch.exp(torch.tensor(avg_loss)).item()),
            "total_loss_tokens": total_loss_tokens,
        },
        "input": {
            "input_ids": input_ids[0].cpu().tolist(),
            "labels": labels[0].cpu().tolist(),
            "loss_mask": loss_mask[0].cpu().tolist(),
            "decoded_text_tokens": prompt_text,
            "text_token_ids": text_token_ids,
        },
        "generation": {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "num_tokens": len(generated_ids),
            "unique_tokens": len(set(generated_ids)),
        },
        "image_tiles": image_info,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n>>> Diagnostics saved to {out_path}")


def _check_weight_stats(model):
    """Print weight statistics for key submodules to verify checkpoint loaded.

    Random init: mean≈0, std≈init_method_std (e.g. 0.0173).
    Trained model: non-trivial mean/std, larger norms.
    """
    checks = [
        ("language_model.embedding", "language_model.embedding"),
        ("language_model.decoder.layers[0]", "language_model.decoder.layers.0"),
        ("language_model.decoder.final_layernorm", "language_model.decoder.final_layernorm"),
        ("language_model.output_layer", "language_model.output_layer"),
        ("vision (radio)", "modality_submodules.images.encoders.radio_encoder"),
        ("projection", "modality_submodules.images.input_projections.0"),
    ]

    for label, attr_path in checks:
        try:
            parts = attr_path.split(".")
            obj = model
            for part in parts:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            params = list(obj.parameters())
            if not params:
                continue
            # Use first non-empty param.
            p = params[0]
            if p.numel() == 0:
                continue
            pf = p.float()
            _print_rank0(f"  {label:40s}: shape={list(p.shape)}, "
                         f"mean={pf.mean().item():.6f}, std={pf.std().item():.6f}, "
                         f"norm={pf.norm().item():.2f}, dtype={p.dtype}")
        except (AttributeError, IndexError, TypeError):
            _print_rank0(f"  {label:40s}: <not found>")


@torch.no_grad()
def _text_only_generation_test(model, tokenizer, args, num_tokens=64):
    """Quick text-only forward pass to check if the LLM backbone is coherent.

    Bypasses the data provider entirely — just tokenizes a prompt, runs the
    MIMO model with no images, and generates greedily.
    """
    model.eval()
    prompt = "What is the capital of France? The capital of France is"

    _print_rank0("\n" + "=" * 70)
    _print_rank0("TEXT-ONLY LLM COHERENCE CHECK")
    _print_rank0("=" * 70)
    _print_rank0(f"  Prompt: {prompt}")

    # Tokenize on all ranks (needed for the forward pass).
    token_ids = tokenizer.tokenize(prompt)
    seq_len = len(token_ids)
    _print_rank0(f"  Tokens: {seq_len} ids = {token_ids}")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # Forward pass without labels → returns logits [B, S, V].
    batch = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": None,
        "loss_mask": None,
    }
    logits, _ = model(**batch)
    _print_rank0(f"  Logits shape: {logits.shape}")

    # Top-5 next token predictions from the last position.
    last_logits = logits[0, -1, :]  # [V]
    top5_vals, top5_ids = torch.topk(last_logits, 5)
    _print_rank0(f"  Top-5 next tokens:")
    for i, (tok_id, val) in enumerate(zip(top5_ids.tolist(), top5_vals.tolist())):
        decoded = tokenizer.detokenize([tok_id])
        _print_rank0(f"    {i+1}. {repr(decoded):20s} (id={tok_id}, logit={val:.2f})")

    # Greedy generation.
    gen_input_ids = input_ids.clone()
    gen_position_ids = position_ids.clone()
    generated_ids = []
    pad_token_id = tokenizer.pad

    for step in range(num_tokens):
        gen_batch = {
            "input_ids": gen_input_ids,
            "position_ids": gen_position_ids,
            "attention_mask": None,
            "labels": None,
            "loss_mask": None,
        }
        step_logits, _ = model(**gen_batch)
        next_token = step_logits[0, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_token)

        if next_token == tokenizer.eod:
            _print_rank0(f"  EOS at step {step + 1}")
            break

        new_len = gen_input_ids.shape[1] + 1
        new_ids = torch.full((1, new_len), pad_token_id, dtype=torch.long, device="cuda")
        new_ids[0, :gen_input_ids.shape[1]] = gen_input_ids[0]
        new_ids[0, -1] = next_token
        gen_input_ids = new_ids
        gen_position_ids = torch.arange(new_len, device="cuda").unsqueeze(0)

    generated_text = tokenizer.detokenize(generated_ids)
    _print_rank0(f"\n  Generated ({len(generated_ids)} tokens):")
    _print_rank0(f"  {prompt}{generated_text}")
    _print_rank0(f"\n  Token IDs: {generated_ids[:30]}{'...' if len(generated_ids) > 30 else ''}")
    unique = len(set(generated_ids))
    _print_rank0(f"  Unique: {unique}/{len(generated_ids)}")
    if unique <= 2 and len(generated_ids) > 10:
        _print_rank0("  >>> DEGENERATE: LLM backbone appears broken or randomly initialized")
    _print_rank0("=" * 70)


def _log_raw_samples(train_dl, tokenizer):
    """Monkey-patch the encoder's batch() to log raw energon samples before expansion.

    This lets us see what energon actually delivers (tokens, labels, num_tiles)
    before tile expansion eats the sequence budget.
    """
    if train_dl is None:
        return  # non-TP-rank-0

    # Walk the dataloader to find the task encoder with a batch() method.
    loader = train_dl
    encoder = None
    # energon wraps things — try common attribute paths.
    for attr in ("dataset", "task_encoder", "_task_encoder"):
        obj = getattr(loader, attr, None)
        if obj is not None and hasattr(obj, "batch"):
            encoder = obj
            break
        # One level deeper.
        if obj is not None:
            for attr2 in ("task_encoder", "_task_encoder"):
                obj2 = getattr(obj, attr2, None)
                if obj2 is not None and hasattr(obj2, "batch"):
                    encoder = obj2
                    break
            if encoder is not None:
                break

    if encoder is None:
        _print_rank0(">>> Could not find task encoder to patch — raw sample logging skipped.")
        return

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    pad_token_id = tokenizer.pad
    original_batch = encoder.batch

    def patched_batch(samples):
        # Log raw sample info before expansion.
        for s_idx, sample in enumerate(samples):
            tokens = sample.tokens
            labels = getattr(sample, "labels", None)
            num_tiles = getattr(sample, "num_tiles", [])
            n_raw = len(tokens)
            n_img_placeholders = (tokens == image_token_id).sum().item() if isinstance(tokens, torch.Tensor) else sum(1 for t in tokens if t == image_token_id)
            n_raw_text = n_raw - n_img_placeholders
            n_labels_valid = 0
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    n_labels_valid = (labels != -100).sum().item()
                else:
                    n_labels_valid = sum(1 for l in labels if l != -100)
            _print_rank0(
                f"  [RAW sample {s_idx}] tokens={n_raw} "
                f"(text={n_raw_text}, img_placeholders={n_img_placeholders}), "
                f"num_tiles={list(num_tiles) if hasattr(num_tiles, '__iter__') else num_tiles}, "
                f"valid_labels={n_labels_valid}/{n_raw}, "
                f"n_images={len(getattr(sample, 'images', []))}"
            )
        return original_batch(samples)

    encoder.batch = patched_batch
    _print_rank0(">>> Patched encoder.batch() to log raw energon samples.")


def main():
    # Initialize Megatron with all the standard args.
    initialize_megatron(
        extra_args_provider=add_generate_args,
        args_defaults={},
    )

    args = get_args()
    tokenizer = get_tokenizer()
    num_generate = getattr(args, "generate_tokens", 128)

    _print_rank0("\n>>> Building model (no optimizer needed for inference)...")

    # Defer checkpoint loading to AFTER get_model so it happens on GPU in bf16.
    # Temporarily clear --nemotron-checkpoint so model_provider skips loading.
    nemotron_ckpt = getattr(args, "nemotron_checkpoint", None)
    args.nemotron_checkpoint = None

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Restore the arg.
    args.nemotron_checkpoint = nemotron_ckpt

    # get_model returns a list of model chunks; take the first.
    model = model[0]

    # Float16Module wraps the actual model under .module
    unwrapped = model.module if hasattr(model, "module") else model

    # ── Weight stats BEFORE loading ──
    _print_rank0("\n>>> Weight stats BEFORE checkpoint load (should look like random init):")
    _check_weight_stats(unwrapped)

    # ── Load checkpoint AFTER model is on GPU in bf16 ──
    if nemotron_ckpt is not None:
        _print_rank0(f"\n>>> Loading nemotron checkpoint from {nemotron_ckpt} ...")
        from utils.model_helpers import load_nemotron_vlm_ckpt
        load_nemotron_vlm_ckpt(unwrapped, nemotron_ckpt)
        _print_rank0(">>> Checkpoint loaded.")

    # ── Weight stats AFTER loading ──
    _print_rank0("\n>>> Weight stats AFTER checkpoint load (should differ from above):")
    _check_weight_stats(unwrapped)

    # Freeze all params — inference only, no gradients needed.
    for p in model.parameters():
        p.requires_grad = False

    # ── Quick text-only sanity check (no vision, no dataloader) ──
    _text_only_generation_test(model, tokenizer, args, num_generate)

    _print_rank0(">>> Model loaded. Fetching one data sample...")

    # Get one batch from the data provider.
    # The provider signature requires train_val_test_num_samples; pass dummy values.
    # Only TP rank 0 gets a real dataloader; others get None (data is broadcast
    # inside get_batch via the TP group).
    train_valid_test_datasets_provider.is_distributed = True
    train_dl, _, _ = train_valid_test_datasets_provider(train_val_test_num_samples=[1, 0, 0])
    data_iter = iter(train_dl) if train_dl is not None else None

    # Move batch to GPU (get_batch may return CPU tensors).
    def _to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, dict):
            return {k: _to_cuda(v) for k, v in obj.items()}
        return obj

    # Derive token IDs from tokenizer (matches remote reference).
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    pad_token_id = tokenizer.pad

    # Monkey-patch the encoder's batch() to log raw energon samples before expansion.
    _log_raw_samples(train_dl, tokenizer)

    # Fetch batches until we find one with BOTH image and text tokens.
    MAX_SKIP = 100
    batch = None
    for attempt in range(MAX_SKIP):
        raw_batch = get_batch(data_iter)
        if raw_batch is None:
            _print_rank0("ERROR: No data returned from dataloader!")
            return
        raw_batch = _to_cuda(raw_batch)
        ids = raw_batch["input_ids"][0]
        n_image = (ids == image_token_id).sum().item()
        n_text = ((ids != image_token_id) & (ids != pad_token_id)).sum().item()
        n_loss = raw_batch["loss_mask"].sum().item()
        has_modality = ("modality_inputs" in raw_batch and raw_batch["modality_inputs"] is not None)
        if n_image > 0 and n_text > 0 and n_loss > 0 and has_modality:
            batch = raw_batch
            _print_rank0(f">>> Found vision+text batch at attempt {attempt}: "
                         f"{n_image} image tokens, {n_text} text tokens, "
                         f"{int(n_loss)} loss tokens.")
            break
        _print_rank0(f">>> Batch {attempt}: img={n_image} text={n_text} loss={int(n_loss)} "
                     f"modality={has_modality} — skipping...")

    if batch is None:
        _print_rank0(f"ERROR: No vision+text batch found after {MAX_SKIP} attempts!")
        return

    _print_rank0(f">>> Batch keys: {list(batch.keys())}")
    _print_rank0(f">>> input_ids shape: {batch['input_ids'].shape}")
    if "modality_inputs" in batch and batch["modality_inputs"] is not None:
        for mod_name, mod_inputs in batch["modality_inputs"].items():
            for enc_name, enc_inputs in mod_inputs.items():
                for inp_name, inp_val in enc_inputs.items():
                    if isinstance(inp_val, torch.Tensor):
                        _print_rank0(
                            f">>> modality_inputs[{mod_name}][{enc_name}][{inp_name}]: "
                            f"{inp_val.shape} {inp_val.dtype}"
                        )

    generate_greedy(
        model,
        batch,
        num_tokens=num_generate,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
    )


if __name__ == "__main__":
    main()
