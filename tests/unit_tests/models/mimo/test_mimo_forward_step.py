# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure (no-GPU) tests for the MIMO stock-schedule forward step + loss_func.

Covers what can be validated without 8 GPUs:

  * ``loss_func`` returns the stock per-token-loss 3-tuple, and critically that
    ``num_tokens`` is an *integer* dtype tensor (a float there casts-faults the
    stock schedule's int ``total_num_tokens`` accumulation and deadlocks);
  * ``loss_func`` raises on a missing / mismatched loss_mask;
  * the ``loss_dict`` carries ``{"lm loss": stack(loss_sum, num_tokens)}`` with
    the expected shape and values;
  * ``move_batch_to_cuda`` recurses through dicts, lists, tuples, and
    ``PackedSeqParams`` tensor fields. We avoid requiring a GPU by recording
    which tensors would be moved via a monkeypatched ``.cuda``.

These run on CPU, so they are safe in the CPU CI lane.
"""

from __future__ import annotations

import torch

from examples.mimo.training.step import loss_func, move_batch_to_cuda
from megatron.core.packed_seq_params import PackedSeqParams


def test_loss_func_returns_int_num_tokens_three_tuple():
    """loss_func returns (loss_sum, num_tokens, loss_dict); num_tokens is int."""
    output = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    loss_sum, num_tokens, loss_dict = loss_func(output, loss_mask=loss_mask)

    # num_tokens MUST be an integer dtype tensor for the stock schedule.
    assert isinstance(num_tokens, torch.Tensor)
    assert not num_tokens.is_floating_point()
    assert num_tokens.dtype in (torch.int32, torch.int64, torch.int16)
    assert int(num_tokens.item()) == 3  # three unmasked tokens

    # loss_sum is the summed (not averaged) masked loss.
    assert isinstance(loss_sum, torch.Tensor)
    assert loss_sum.shape == torch.Size([])
    assert torch.allclose(loss_sum, torch.tensor(1.0 + 2.0 + 4.0))

    # loss_dict matches the prototype's logging contract.
    assert set(loss_dict.keys()) == {"lm loss"}
    logged = loss_dict["lm loss"]
    assert logged.shape == torch.Size([2])
    assert torch.allclose(logged[0], loss_sum.detach())
    assert torch.allclose(logged[1], num_tokens.detach().float())


def test_loss_func_missing_mask_raises():
    output = torch.tensor([[1.0, 2.0]])
    try:
        loss_func(output, loss_mask=None)
    except RuntimeError as exc:
        assert "loss_mask" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on missing loss_mask")


def test_loss_func_shape_mismatch_raises():
    output = torch.tensor([[1.0, 2.0, 3.0]])
    loss_mask = torch.tensor([[1.0, 1.0]])
    try:
        loss_func(output, loss_mask=loss_mask)
    except RuntimeError as exc:
        assert "does not match loss_mask shape" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on shape mismatch")


def test_loss_func_none_output_raises():
    try:
        loss_func(None, loss_mask=torch.tensor([[1.0]]))
    except RuntimeError as exc:
        assert "no loss tensor" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on None output")


def test_loss_func_non_tensor_output_raises():
    try:
        loss_func([1.0, 2.0], loss_mask=torch.tensor([[1.0]]))
    except TypeError as exc:
        assert "expects the terminal language stage to return a tensor" in str(exc)
    else:
        raise AssertionError("expected TypeError on non-tensor output")


def _track_cuda(monkeypatch):
    """Monkeypatch Tensor.cuda to record moves without needing a GPU.

    Returns the set that accumulates ids of tensors asked to move. ``.cuda``
    returns the same tensor object (id-stable) so the structure is preserved
    and identity assertions hold. We do NOT patch ``is_cuda``: CPU tensors
    already report ``is_cuda == False``, so the PackedSeqParams ``not is_cuda``
    gate naturally selects them, and ``torch.Tensor.is_cuda`` is a built-in
    getset descriptor that cannot be reassigned on the C type anyway.
    """
    moved: set[int] = set()

    def fake_cuda(self, *args, **kwargs):
        moved.add(id(self))
        return self

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda)
    return moved


def test_move_batch_to_cuda_recurses_dict_list_tuple(monkeypatch):
    moved = _track_cuda(monkeypatch)

    t_top = torch.tensor([1.0])
    t_in_list = torch.tensor([2.0])
    t_in_tuple = torch.tensor([3.0])
    t_nested = torch.tensor([4.0])

    batch = {
        "input_ids": t_top,
        "a_list": [t_in_list, "not a tensor", 7],
        "a_tuple": (t_in_tuple,),
        "nested": {"deep": t_nested},
        "scalar": 5,
    }

    out = move_batch_to_cuda(batch)

    # Structure is preserved.
    assert isinstance(out, dict)
    assert isinstance(out["a_list"], list)
    assert isinstance(out["a_tuple"], tuple)
    assert out["scalar"] == 5
    assert out["a_list"][1] == "not a tensor"

    # Every tensor leaf was asked to move to cuda.
    for t in (t_top, t_in_list, t_in_tuple, t_nested):
        assert id(t) in moved


def test_move_batch_to_cuda_handles_packed_seq_params(monkeypatch):
    moved = _track_cuda(monkeypatch)

    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_kv = torch.tensor([0, 4, 8], dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=8,  # int, not a tensor -> left alone
        max_seqlen_kv=8,
    )

    batch = {"packing": psp}
    out = move_batch_to_cuda(batch)

    # Same dataclass instance is returned (in-place field updates).
    assert out["packing"] is psp
    # qkv_format (a str) and int max_seqlen fields are untouched.
    assert psp.qkv_format == "thd"
    assert psp.max_seqlen_q == 8
    # Tensor fields were asked to move.
    assert id(cu_q) in moved
    assert id(cu_kv) in moved


def test_move_batch_to_cuda_passes_through_non_tensor():
    # No monkeypatch: plain Python leaves are returned unchanged.
    assert move_batch_to_cuda(None) is None
    assert move_batch_to_cuda(42) == 42
    assert move_batch_to_cuda("x") == "x"
