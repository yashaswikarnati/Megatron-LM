"""Standalone encoder-only smoke test for the routed-iter multi-lane merge.

Builds an 8-rank encoder grid (TP=1 CP=1 PP=1 DP=8 EP=1), forces virtual
``llm_dp=32`` so ``lanes_per_encoder=4`` (the exact fan-out that breaks 9n
GBS=192), drives the production routed-iter code path (``_build_routed_
encoder_iterator`` + ``_combine_encoder_batches`` + ``_concat_packed_seq_
params``), and runs RADIO forward only. No LLM, no bridge, no DDP overlap.

Each step prints the merged ``cu_seqlens_q`` / ``max_seqlen_q`` /
``images.shape`` so a malformed merge surfaces as a value-level anomaly
instead of a 10-minute NCCL hang.
"""

import argparse
import os
import sys
import time
import traceback
from datetime import timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist


def _print(rank: int, msg: str) -> None:
    print(f"[r{rank}] {msg}", flush=True)


def _parse_args() -> argparse.Namespace:
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        NEMOTRON_54L_MODEL_PROVIDER,
        add_model_provider_args,
        prepare_model_provider_args,
    )

    p = argparse.ArgumentParser()
    # Inherit the full model-provider arg surface so every attribute the
    # production code path reads off args is present with the right default.
    add_model_provider_args(p)
    # test-specific knobs (a few overlap with the provider group but argparse
    # tolerates dest collisions on identical names)
    p.add_argument("--data-path", required=True)
    p.add_argument("--encoder-dp", type=int, default=8)
    p.add_argument("--llm-dp", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--shuffle-buffer-size", type=int, default=100)
    p.add_argument("--packing-buffer-size", type=int, default=4)
    p.add_argument("--max-samples-per-sequence", type=int, default=100)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument(
        "--parity",
        action="store_true",
        help="Compare merged encoder forward vs per-lane forwards (numerics check).",
    )
    p.add_argument("--parity-atol", type=float, default=2e-3)
    p.add_argument("--parity-rtol", type=float, default=1e-2)
    _ = NEMOTRON_54L_MODEL_PROVIDER  # silence import-unused; provider is selected via --model-provider
    args = p.parse_args()
    # Apply the same provider-default patching the production loop runs after
    # parse_args (sets seq_length, image_seq_length, hidden_size, num_layers,
    # vision_encoder_key, vision_input_mode, ...).
    prepare_model_provider_args(args)
    return args


def main() -> None:
    args = _parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(args.encoder_dp)))
    if world != args.encoder_dp:
        raise RuntimeError(
            f"WORLD_SIZE={world} must equal --encoder-dp={args.encoder_dp}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    _print(rank, f"init done world={world} local_rank={local_rank}")

    from examples.mimo.data import hetero_energon as _he
    from examples.mimo.data.hetero_energon import (
        _build_routed_encoder_iterator,
        _llm_lanes_for_encoder_rank,
    )
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        RADIOEncoderWrapper,
        radio_vision_config,
    )
    from examples.mimo.training.hetero.step import move_batch_to_cuda
    from examples.mimo.training.hetero.topology import (
        create_hypercomm_grid,
        get_pg_collection,
    )
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec,
    )

    encoder_grid = create_hypercomm_grid(
        offset=0,
        tp=1,
        cp=1,
        pp=1,
        dp=args.encoder_dp,
        ep=1,
        expt_tp=None,
        expt_dp=None,
    )
    pg_coll = get_pg_collection(encoder_grid)
    tp_pg = encoder_grid.get_pg("tp")
    encoder_dp_rank = rank

    llm_lanes = _llm_lanes_for_encoder_rank(args, encoder_dp_rank)
    _print(
        rank,
        f"llm_lanes={llm_lanes} lanes_per_encoder={len(llm_lanes)}",
    )

    iterator = _build_routed_encoder_iterator(
        args,
        tp_group=tp_pg,
        encoder_dp_rank=encoder_dp_rank,
        llm_lanes=llm_lanes,
    )

    vision_config = radio_vision_config(args, tp_size=1, pp_size=1)
    encoder = RADIOEncoderWrapper(
        transformer_config=vision_config,
        transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
        pg_collection=pg_coll,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        class_token_len=args.class_token_len,
        drop_class_token=True,
        apply_pixel_shuffle=True,
        force_eval_mode=args.freeze_vit,
        dynamic_resolution=args.dynamic_resolution,
    ).cuda()
    encoder.eval()
    _print(rank, f"encoder built dtype={next(encoder.parameters()).dtype}")

    # Parity mode: install a module-level capture hook on
    # _combine_encoder_batches so we can pull both the per-lane batches and
    # the merged batch from a single iterator step (the routed iterator's
    # ``next_encoder_batch`` resolves _combine_encoder_batches by module-global
    # name at call time, so patching the attribute on _he intercepts cleanly).
    captured: dict = {"lane_batches": None}
    if args.parity:
        _real_combine = _he._combine_encoder_batches

        def _capturing_combine(batches):
            captured["lane_batches"] = list(batches)
            return _real_combine(batches)

        _he._combine_encoder_batches = _capturing_combine
        _print(rank, "parity mode ON: capturing per-lane batches for comparison")

    dist.barrier()

    for step in range(args.n_steps):
        t0 = time.time()
        try:
            batch = next(iterator)
        except Exception as e:
            _print(rank, f"step={step} iterator FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        mi = batch.get("modality_inputs") if isinstance(batch, dict) else None
        if mi is None:
            _print(rank, f"step={step} no modality_inputs (text-only batch), skip")
            continue
        enc_inputs = mi["images"]["radio_encoder"]
        psp = enc_inputs.get("packed_seq_params")
        if psp is not None:
            cu = psp.cu_seqlens_q
            mono = bool(torch.all(cu[1:] >= cu[:-1]).item())
            _print(
                rank,
                f"step={step} cu_seqlens_q={cu.tolist()} "
                f"max_seqlen_q={int(psp.max_seqlen_q)} "
                f"monotonic={mono} "
                f"images.shape={tuple(enc_inputs['x'].shape)} "
                f"imgs_sizes={enc_inputs.get('imgs_sizes')}",
            )
            if not mono:
                raise RuntimeError(
                    f"non-monotonic cu_seqlens_q at step={step}: {cu.tolist()}"
                )
        else:
            _print(
                rank,
                f"step={step} no packed_seq_params; "
                f"x.shape={tuple(enc_inputs['x'].shape)}",
            )

        enc_inputs_cuda = move_batch_to_cuda(enc_inputs)

        t1 = time.time()
        try:
            with torch.no_grad():
                out = encoder(**enc_inputs_cuda)
        except Exception as e:
            _print(rank, f"step={step} forward FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        torch.cuda.synchronize()
        t2 = time.time()
        _print(
            rank,
            f"step={step} OK out_shape={tuple(out.shape)} "
            f"t_iter={t1 - t0:.2f}s t_fwd={t2 - t1:.2f}s",
        )

        if args.parity and captured["lane_batches"] is not None:
            lane_batches = captured["lane_batches"]
            captured["lane_batches"] = None  # reset for next step
            n_with_images = sum(
                1 for lb in lane_batches if lb.get("modality_inputs") is not None
            )
            if n_with_images < 2:
                _print(
                    rank,
                    f"step={step} parity: only {n_with_images} lane(s) with images; "
                    "merge degenerates to identity, skipping comparison",
                )
                continue

            out_lanes = []
            for k, lb in enumerate(lane_batches):
                if lb.get("modality_inputs") is None:
                    out_lanes.append(None)
                    continue
                lane_inputs = lb["modality_inputs"]["images"]["radio_encoder"]
                lane_inputs_cuda = move_batch_to_cuda(lane_inputs)
                with torch.no_grad():
                    out_lane = encoder(**lane_inputs_cuda)
                out_lanes.append(out_lane)
            torch.cuda.synchronize()

            offset = 0
            worst_max_diff = 0.0
            worst_cos = 1.0
            for k, ol in enumerate(out_lanes):
                if ol is None:
                    continue
                length = ol.shape[1]
                slice_k = out[:, offset : offset + length, :]
                offset += length
                if slice_k.shape != ol.shape:
                    raise RuntimeError(
                        f"step={step} lane={k} shape mismatch: "
                        f"slice={tuple(slice_k.shape)} ref={tuple(ol.shape)}"
                    )
                max_diff = (slice_k - ol).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    slice_k.float().flatten().unsqueeze(0),
                    ol.float().flatten().unsqueeze(0),
                    dim=1,
                ).item()
                worst_max_diff = max(worst_max_diff, max_diff)
                worst_cos = min(worst_cos, cos)
                ok = torch.allclose(slice_k, ol, atol=args.parity_atol, rtol=args.parity_rtol)
                tag = "OK" if ok else "MISMATCH"
                _print(
                    rank,
                    f"step={step} lane={k} {tag} "
                    f"len={length} max_diff={max_diff:.4e} cos={cos:.6f}",
                )
                if not ok:
                    raise RuntimeError(
                        f"parity FAIL step={step} lane={k}: "
                        f"max_diff={max_diff:.4e} > atol={args.parity_atol}"
                    )
            if offset != out.shape[1]:
                raise RuntimeError(
                    f"step={step} sliced {offset} tokens but merged output has "
                    f"{out.shape[1]} — slice math is wrong"
                )
            _print(
                rank,
                f"step={step} parity PASS (n_lanes_with_images={n_with_images}) "
                f"worst_max_diff={worst_max_diff:.4e} worst_cos={worst_cos:.6f}",
            )

    dist.barrier()
    _print(rank, "all steps done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
