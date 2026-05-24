# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import random
from argparse import Namespace
from collections import deque

import pytest
import torch

from examples.mimo.data import hetero_energon
from examples.mimo.data.energon_multimodal_provider import MimoMultiModalPackingEncoder
from examples.mimo.data.hetero_energon import EnergonIterator, _route_samples_to_lanes
from examples.mimo.training.hetero.checkpointing import (
    maybe_restore_dataloader_state,
    maybe_save_dataloader_state,
)

_PROVENANCE_KEY = MimoMultiModalPackingEncoder.PROVENANCE_KEY


class RandomLoader:
    """Small dataloader stub that consumes Python's global random module."""

    def __iter__(self):
        return self

    def __next__(self):
        return {"value": random.randrange(1_000_000)}


class SavableListLoader:
    """Small savable-loader stub with Energon's save/restore method names."""

    def __init__(self, samples):
        self.samples = list(samples)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration
        sample = dict(self.samples[self.index])
        self.index += 1
        return sample

    def save_state_rank(self):
        return {"index": self.index}

    def restore_state_rank(self, state):
        self.index = state["index"]


def test_energon_iterator_uses_isolated_python_random_state():
    """Same DP-lane iterators should match without perturbing caller RNG state."""
    first = EnergonIterator(RandomLoader(), random_seed=12345)
    second = EnergonIterator(RandomLoader(), random_seed=12345)

    first_values = []
    second_values = []
    for _ in range(8):
        random.seed(111)
        caller_state = random.getstate()
        first_values.append(next(first)["value"])
        assert random.getstate() == caller_state

        random.seed(222)
        caller_state = random.getstate()
        second_values.append(next(second)["value"])
        assert random.getstate() == caller_state

    assert first_values == second_values
    assert len(set(first_values)) > 1


def test_energon_iterator_restores_single_lane_loader_state():
    """Restored single-lane iterators continue from the saved loader offset."""
    loader = SavableListLoader([{"value": 10}, {"value": 11}, {"value": 12}])
    iterator = EnergonIterator(
        loader,
        random_seed=123,
        alignment_role="llm",
        llm_lanes=[0],
        state_name="train_dataloader_llm_lane000.pt",
    )

    assert next(iterator)["value"] == 10
    state = iterator.save_state()
    assert next(iterator)["value"] == 11

    restored = EnergonIterator(
        SavableListLoader([{"value": 10}, {"value": 11}, {"value": 12}]),
        random_seed=123,
        alignment_role="llm",
        llm_lanes=[0],
        state_name="train_dataloader_llm_lane000.pt",
    )
    restored.restore_state(state)

    assert next(restored)["value"] == 11


def test_combine_encoder_batches_drops_packing_and_concatenates_modalities():
    """Encoder fan-out combines whole packed samples without carrying LLM packing metadata."""
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, 4]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "packing_kwargs": {"cu_seqlens_q": torch.tensor([0, 3])},
        "modality_inputs": {"images": {"radio_encoder": {"x": torch.ones(1, 3, 4, 4)}}},
    }
    second = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "labels": torch.tensor([[6, 7, 8]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "packing_kwargs": {"cu_seqlens_q": torch.tensor([0, 3])},
        "modality_inputs": {"images": {"radio_encoder": {"x": torch.zeros(2, 3, 4, 4)}}},
    }

    combined = hetero_energon._combine_encoder_batches([first, second])

    assert "packing_kwargs" not in combined
    assert combined["input_ids"].tolist() == [[1, 2, 3], [5, 6, 7]]
    images = combined["modality_inputs"]["images"]["radio_encoder"]["x"]
    assert images.shape == (3, 3, 4, 4)
    assert torch.all(images[:1] == 1)
    assert torch.all(images[1:] == 0)


# ---------------------------------------------------------------------------
# Routed encoder iterator — _route_samples_to_lanes tests
# ---------------------------------------------------------------------------


def _stamped(worker_id: int, payload: object) -> dict:
    """Build a fake encoder batch carrying a provenance stamp."""
    return {_PROVENANCE_KEY: worker_id, "payload": payload}


def _make_loader(samples):
    """Wrap a list of pre-stamped samples in an iterator the routing code can consume."""
    return iter(samples)


def test_route_samples_to_lanes_round_robin_assigns_workers_to_lanes():
    """Workers 0..NW-1 feed lane 0, NW..2NW-1 feed lane 1, etc."""
    # encoder_dp_rank=0, world hosts lanes 0..3 (lane_offset=0).
    # num_workers_per_lane=2 → workers [0,1]→lane0, [2,3]→lane1, [4,5]→lane2, [6,7]→lane3.
    samples = [_stamped(w, f"w{w}") for w in (0, 2, 4, 6)]
    pending = [deque() for _ in range(4)]
    lane_batches, pulls = _route_samples_to_lanes(
        _make_loader(samples),
        lanes_per_encoder=4,
        lane_offset=0,
        num_workers_per_lane=2,
        encoder_dp_rank=0,
        pending_by_lane=pending,
        max_pulls_per_step=16,
        provenance_key=_PROVENANCE_KEY,
    )
    assert pulls == 4
    assert [b["payload"] for b in lane_batches] == ["w0", "w2", "w4", "w6"]
    assert all(len(q) == 0 for q in pending)


def test_route_samples_to_lanes_surplus_lands_in_pending_fifo():
    """A second sample for an already-filled lane is queued for next step."""
    # 2 lanes, NW=2: workers 0,1→lane0; 2,3→lane1.
    # Loader yields w0 (lane0), w1 (lane0 surplus), w2 (lane1) — first step fills.
    samples = [_stamped(0, "a"), _stamped(1, "b"), _stamped(2, "c")]
    pending = [deque() for _ in range(2)]
    lane_batches, pulls = _route_samples_to_lanes(
        _make_loader(samples),
        lanes_per_encoder=2,
        lane_offset=0,
        num_workers_per_lane=2,
        encoder_dp_rank=0,
        pending_by_lane=pending,
        max_pulls_per_step=8,
        provenance_key=_PROVENANCE_KEY,
    )
    assert pulls == 3
    assert [b["payload"] for b in lane_batches] == ["a", "c"]
    assert len(pending[0]) == 1
    assert pending[0][0]["payload"] == "b"
    assert len(pending[1]) == 0


def test_route_samples_to_lanes_drains_pending_before_pulling():
    """Pending lane-0 sample is consumed first; loader is only pulled for empty lanes."""
    pending = [deque([_stamped(0, "stashed")]), deque()]
    # Loader has one new sample for lane 1.
    samples = [_stamped(2, "fresh")]
    lane_batches, pulls = _route_samples_to_lanes(
        _make_loader(samples),
        lanes_per_encoder=2,
        lane_offset=0,
        num_workers_per_lane=2,
        encoder_dp_rank=0,
        pending_by_lane=pending,
        max_pulls_per_step=8,
        provenance_key=_PROVENANCE_KEY,
    )
    assert pulls == 1
    assert [b["payload"] for b in lane_batches] == ["stashed", "fresh"]
    assert len(pending[0]) == 0


def test_routed_iterator_restores_pending_samples_before_pulling():
    """Routed resume must keep samples already pulled ahead of the loader state."""

    def make_iterator():
        loader = SavableListLoader(
            [_stamped(0, "a"), _stamped(1, "b"), _stamped(2, "c"), _stamped(3, "d")]
        )
        loader_iter_holder = [iter(loader)]
        pending = [deque() for _ in range(2)]

        def next_batch():
            lane_batches, _ = _route_samples_to_lanes(
                loader_iter_holder[0],
                lanes_per_encoder=2,
                lane_offset=0,
                num_workers_per_lane=2,
                encoder_dp_rank=0,
                pending_by_lane=pending,
                max_pulls_per_step=8,
                provenance_key=_PROVENANCE_KEY,
            )
            return {"payload": [batch["payload"] for batch in lane_batches]}

        def save_extra_state():
            return {"pending_by_lane": [list(queue) for queue in pending]}

        def restore_extra_state(state):
            for queue, values in zip(pending, state["pending_by_lane"]):
                queue.clear()
                queue.extend(values)
            loader_iter_holder[0] = iter(loader)

        iterator = EnergonIterator(
            loader,
            source_rank=True,
            local_batch_fn=next_batch,
            alignment_role="encoder",
            llm_lanes=[0, 1],
            state_name="train_dataloader_encoder_dprank000_lanes000-001.pt",
            extra_state_fn=save_extra_state,
            restore_extra_state_fn=restore_extra_state,
        )
        return iterator

    iterator = make_iterator()
    assert next(iterator)["payload"] == ["a", "c"]
    state = iterator.save_state()
    assert next(iterator)["payload"] == ["b", "d"]

    restored = make_iterator()
    restored.restore_state(state)

    assert next(restored)["payload"] == ["b", "d"]


def test_checkpoint_helpers_roundtrip_iterator_state(tmp_path):
    """Checkpoint helpers save under iter_NNNNNNN and restore the iterator state."""
    samples = [{"value": 10}, {"value": 11}, {"value": 12}]
    iterator = EnergonIterator(
        SavableListLoader(samples),
        alignment_role="llm",
        llm_lanes=[0],
        state_name="train_dataloader_llm_lane000.pt",
    )
    assert next(iterator)["value"] == 10

    args = Namespace(dataloader_save=str(tmp_path), dataloader_load=None)
    maybe_save_dataloader_state(iterator, 15, args)

    state_path = tmp_path / "iter_0000015" / "train_dataloader_llm_lane000.pt"
    assert state_path.exists()

    restored = EnergonIterator(
        SavableListLoader(samples),
        alignment_role="llm",
        llm_lanes=[0],
        state_name="train_dataloader_llm_lane000.pt",
    )
    maybe_restore_dataloader_state(restored, 15, args)

    assert next(restored)["value"] == 11


def test_route_samples_to_lanes_lane_offset_shifts_global_lane():
    """Encoder rank E>0 owns a non-zero lane_offset; routing subtracts it correctly."""
    # encoder_dp_rank=1, encoder_dp=2, llm_dp=4, NW=1 → lane_offset=2, lanes 2,3 local.
    # Workers in this encoder: global ids 2,3 (E*NW*k + W where k=2, NW=1 → 2 + W).
    # global_llm_lane = global_worker_id // NW = 2,3 → local_lane = 0,1.
    samples = [_stamped(2, "L2"), _stamped(3, "L3")]
    pending = [deque() for _ in range(2)]
    lane_batches, _ = _route_samples_to_lanes(
        _make_loader(samples),
        lanes_per_encoder=2,
        lane_offset=2,
        num_workers_per_lane=1,
        encoder_dp_rank=1,
        pending_by_lane=pending,
        max_pulls_per_step=8,
        provenance_key=_PROVENANCE_KEY,
    )
    assert [b["payload"] for b in lane_batches] == ["L2", "L3"]


def test_route_samples_to_lanes_raises_on_pull_budget_exhaustion():
    """When the loader can't fill every lane in the budget, fail loudly."""
    # 2 lanes but loader only delivers to lane 0.
    samples = [_stamped(0, "x"), _stamped(0, "y"), _stamped(1, "z")]
    pending = [deque() for _ in range(2)]
    with pytest.raises(RuntimeError, match="did not yield samples for local_lanes"):
        _route_samples_to_lanes(
            _make_loader(samples),
            lanes_per_encoder=2,
            lane_offset=0,
            num_workers_per_lane=2,
            encoder_dp_rank=0,
            pending_by_lane=pending,
            max_pulls_per_step=3,
            provenance_key=_PROVENANCE_KEY,
        )


def test_route_samples_to_lanes_raises_on_out_of_range_worker():
    """A worker id from a foreign encoder rank surfaces as a hard error."""
    # encoder_dp_rank=0 owns lanes 0..1 with NW=2, so global_worker_id 0..3 are valid.
    # A stray sample stamped with worker 4 (which belongs to rank 1) should fail.
    samples = [_stamped(0, "ok"), _stamped(4, "stray")]
    pending = [deque() for _ in range(2)]
    with pytest.raises(RuntimeError, match="outside encoder rank"):
        _route_samples_to_lanes(
            _make_loader(samples),
            lanes_per_encoder=2,
            lane_offset=0,
            num_workers_per_lane=2,
            encoder_dp_rank=0,
            pending_by_lane=pending,
            max_pulls_per_step=8,
            provenance_key=_PROVENANCE_KEY,
        )


def test_route_samples_to_lanes_raises_when_provenance_missing():
    """Samples without a provenance stamp fail with a clear message."""
    samples = [{"payload": "missing"}]
    pending = [deque()]
    with pytest.raises(RuntimeError, match="attach_provenance"):
        _route_samples_to_lanes(
            _make_loader(samples),
            lanes_per_encoder=1,
            lane_offset=0,
            num_workers_per_lane=1,
            encoder_dp_rank=0,
            pending_by_lane=pending,
            max_pulls_per_step=4,
            provenance_key=_PROVENANCE_KEY,
        )


# ---------------------------------------------------------------------------
# Bit-wise parity: routed iterator must produce the same per-lane sample
# sequence as the previous per-lane iterators would have.
# ---------------------------------------------------------------------------


def test_routed_iterator_matches_per_lane_global_worker_ids():
    """For the (rank, world_size, num_workers) reshape used by the routed iterator,
    the producing global_worker_id at the encoder side equals the per-lane
    global_worker_id, lane-by-lane, sample-by-sample.

    This is the algebraic property that gives bit-wise sample parity with the
    previous multi-iterator path. ``megatron.energon.worker.WorkerConfig.worker_seed``
    hashes only ``global_worker_id`` and ``seed_offset``, and
    ``WebdatasetSharder.split_samples_to_workers`` partitions shards by global
    worker index over ``global_workers = world_size * num_workers``, so equal
    global_worker_ids ⇒ equal shards ⇒ equal samples in equal order.
    """
    # llm_dp=8, encoder_dp=2 → lanes_per_encoder=4, NW=2 per lane.
    llm_dp = 8
    encoder_dp = 2
    num_workers_per_lane = 2
    lanes_per_encoder = llm_dp // encoder_dp
    num_workers_enc = num_workers_per_lane * lanes_per_encoder

    # OLD scheme: for lane L, the workers have global_worker_ids
    #   L*NW + w  for w in [0, NW).
    old_by_lane = {
        lane: [lane * num_workers_per_lane + w for w in range(num_workers_per_lane)]
        for lane in range(llm_dp)
    }

    # NEW scheme: for encoder rank E, worker W → global_worker_id = E*num_workers_enc + W,
    # routed to local_lane = (global_worker_id // NW) - lane_offset (= W // NW).
    for encoder_dp_rank in range(encoder_dp):
        lane_offset = encoder_dp_rank * lanes_per_encoder
        new_by_local_lane: dict[int, list[int]] = {lane: [] for lane in range(lanes_per_encoder)}
        for W in range(num_workers_enc):
            gid_new = encoder_dp_rank * num_workers_enc + W
            local_lane = (gid_new // num_workers_per_lane) - lane_offset
            new_by_local_lane[local_lane].append(gid_new)

        for local_lane in range(lanes_per_encoder):
            global_lane = lane_offset + local_lane
            assert new_by_local_lane[local_lane] == old_by_lane[global_lane], (
                f"global_worker_id mismatch at encoder_dp_rank={encoder_dp_rank}, "
                f"local_lane={local_lane}: new={new_by_local_lane[local_lane]} "
                f"vs old={old_by_lane[global_lane]}"
            )


def test_routed_iterator_preserves_global_workers_invariant():
    """The reshape preserves the total global worker count, which is what makes
    Energon's per-worker shard partitioning identical between the per-lane and
    routed configurations (see split_samples_to_workers)."""
    for llm_dp, encoder_dp, num_workers in [(8, 2, 2), (16, 1, 4), (32, 8, 2), (128, 16, 4)]:
        lanes_per_encoder = llm_dp // encoder_dp
        old_global_workers = llm_dp * num_workers
        new_global_workers = encoder_dp * (num_workers * lanes_per_encoder)
        assert old_global_workers == new_global_workers, (
            f"global_workers diverged for llm_dp={llm_dp} encoder_dp={encoder_dp}: "
            f"old={old_global_workers} new={new_global_workers}"
        )
