# Heterogeneous Training-Loop Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `schedule_pg_collection` from the outer training API and carry one union-typed `pg_collection` plus its matching communicator through pretrain, train, train-step, logging, and evaluation.

**Architecture:** Orchestration retains `ProcessGroupCollection | MultiModuleProcessGroupCollection` until an operation names the module or policy role it needs. Plain runs keep their existing MPU fallback and model-parallel reductions. Multi-module runs resolve only the loss module for loss-owned operations, trust the heterogeneous optimizer's already-coherent step statistics, and require a `MultiModulePipelineCommunicator` before schedule execution.

**Tech Stack:** Python, PyTorch distributed, Megatron Core training loop, pytest, 8-rank `torch.distributed.run`, Cog/Slurm.

---

## File Structure

- `megatron/training/training.py`: union types, carrier/communicator validation, pretrain/train/train-step/evaluation/logging plumbing.
- `examples/mimo/pretrain_mimo.py`: pass the topology carrier through the sole `pg_collection` keyword.
- `megatron/training/models/base.py`: describe the builder orchestration boundary as accepting the union without changing ordinary builder behavior.
- `tests/unit_tests/training/test_train_step_schedule_plumbing.py`: identity forwarding, pair validation, and plain-vs-multi optimizer/loss semantics.
- `tests/unit_tests/training/test_pg_collection_plumbing.py`: pretrain/train signature and exact-object plumbing tests.
- `tests/unit_tests/training/test_evaluation_pg_collection.py`: heterogeneous evaluation, loss ownership, and non-loss participation tests.
- `tests/unit_tests/training/test_training_log_pg_collection.py`: carrier-safe learning-rate, MoE, and memory reporting tests.

Checkpoint group extraction is intentionally excluded from this plan. During this slice, checkpoint-disabled train/eval is the executable acceptance path. The next plan adds the named single-transport checkpoint adapter before save/resume acceptance.

### Task 1: Establish One Outer API and Validate Schedule Pairs

**Files:**
- Modify: `megatron/training/training.py:1004-1115,3245-3268`
- Modify: `examples/mimo/pretrain_mimo.py:137-146`
- Modify: `megatron/training/models/base.py` at `build_distributed_models`
- Create: `tests/unit_tests/training/test_pg_collection_plumbing.py`
- Modify: `tests/unit_tests/training/test_train_step_schedule_plumbing.py`

- [ ] **Step 1: Write failing signature and pair tests**

Add signature tests:

```python
import inspect


def test_training_entrypoints_expose_only_pg_collection():
    for entrypoint in (training_mod.pretrain, training_mod.train, training_mod.train_step):
        parameters = inspect.signature(entrypoint).parameters
        assert "pg_collection" in parameters
        assert "schedule_pg_collection" not in parameters
```

Add focused tests for a private pair resolver:

```python
def test_multimodule_carrier_requires_multimodule_communicator():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection()},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    with pytest.raises(ValueError, match="MultiModulePipelineCommunicator"):
        training_mod._resolve_pipeline_communicator(carrier, None, SimpleNamespace())


def test_plain_carrier_builds_ordinary_communicator(mocker):
    carrier = ProcessGroupCollection(pp="pp")
    communicator_cls = mocker.patch.object(training_mod, "P2PCommunicator")
    config = SimpleNamespace()

    communicator = training_mod._resolve_pipeline_communicator(carrier, None, config)

    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(pp_group="pp", config=config)
```

Also cover accepted plain/ordinary and multi/multi pairs, plus both mismatched pair directions.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit_tests/training/test_pg_collection_plumbing.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
```

Expected: signatures still expose `schedule_pg_collection`, and the resolver is absent.

- [ ] **Step 3: Add generic orchestration types and the pair resolver**

In `training.py`, import `MultiModulePipelineCommunicator` and define:

```python
PGCollection = Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
PipelineCommunicator = Union[P2PCommunicator, MultiModulePipelineCommunicator]


def _resolve_pipeline_communicator(
    pg_collection: Optional[PGCollection],
    p2p_communicator: Optional[PipelineCommunicator],
    config,
) -> Optional[PipelineCommunicator]:
    if isinstance(pg_collection, MultiModuleProcessGroupCollection):
        if not isinstance(p2p_communicator, MultiModulePipelineCommunicator):
            raise ValueError(
                "MultiModuleProcessGroupCollection requires a "
                "MultiModulePipelineCommunicator"
            )
        return p2p_communicator
    if isinstance(p2p_communicator, MultiModulePipelineCommunicator):
        raise ValueError(
            "MultiModulePipelineCommunicator requires a "
            "MultiModuleProcessGroupCollection"
        )
    if isinstance(pg_collection, ProcessGroupCollection) and p2p_communicator is None:
        return P2PCommunicator(pp_group=pg_collection.pp, config=config)
    if pg_collection is None and p2p_communicator is not None:
        raise ValueError("An explicit communicator requires an explicit pg_collection")
    return p2p_communicator
```

This helper performs no collectives and must run before schedule selection.

- [ ] **Step 4: Replace the outer alias in pretrain**

Use this parameter shape:

```python
p2p_communicator: Optional[PipelineCommunicator] = None,
pg_collection: Optional[PGCollection] = None,
```

Delete `init_pg_collection`. Before `initialize_megatron`, narrow only a plain carrier for bootstrap seed groups:

```python
bootstrap_pg_collection = (
    pg_collection if isinstance(pg_collection, ProcessGroupCollection) else None
)
```

Pass groups from `bootstrap_pg_collection`; a multi-module carrier supplies no representative seed group. After initialization:

```python
if pg_collection is None:
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
```

Keep that exact object through setup, train, final validation/test, and later checkpoint routing. JIT TP size is taken only from `bootstrap_pg_collection`; otherwise it remains `None`.

- [ ] **Step 5: Rename train and the MIMO caller**

Change `train(..., pg_collection: Optional[PGCollection] = None)` and delete its alias documentation. Change `examples/mimo/pretrain_mimo.py` to:

```python
pretrain(
    ...,
    p2p_communicator=communicator,
    pg_collection=topology.pg_collection,
)
```

Widen the base builder annotation to the union. Do not add runtime MIMO logic to the base class.

- [ ] **Step 6: Run tests, format, and commit**

```bash
uv run isort megatron/training/training.py examples/mimo/pretrain_mimo.py \
  megatron/training/models/base.py tests/unit_tests/training/test_pg_collection_plumbing.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
uv run pytest -q tests/unit_tests/training/test_pg_collection_plumbing.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
git diff --check
git add megatron/training/training.py examples/mimo/pretrain_mimo.py \
  megatron/training/models/base.py tests/unit_tests/training/test_pg_collection_plumbing.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
git commit -s -m "Unify heterogeneous training process group API"
```

Expected: focused tests pass; local execution may stop before collection when Torch is absent, in which case record the exact failure and use Cog for the verdict.

### Task 2: Make Train Step Union-Safe

**Files:**
- Modify: `megatron/training/training.py:2278-2493`
- Modify: `tests/unit_tests/training/test_train_step_schedule_plumbing.py`

- [ ] **Step 1: Write failing forwarding and reduction tests**

Replace the old plumbing assertion with exact identity:

```python
def test_train_step_forwards_the_same_carrier_and_communicator():
    carrier = _multi_carrier(loss_local=False)
    communicator = mock.create_autospec(MultiModulePipelineCommunicator, instance=True)

    captured = _run(pg_collection=carrier, p2p_communicator=communicator)

    assert captured["pg_collection"] is carrier
    assert captured["p2p_communicator"] is communicator
```

Add a full-step harness whose optimizer returns fixed values. Patch the generic MP reduction helpers to raise for the multi case and assert values are returned unchanged. For the plain case, assert both helpers receive `plain.mp`.

Add loss ownership tests:

```python
def test_encoder_only_train_step_does_not_read_plain_groups(mocker):
    carrier = _multi_carrier(loss_local=False)
    mocker.patch.object(
        training_mod.ProcessGroupCollection,
        "use_mpu_process_groups",
        side_effect=AssertionError("MPU fallback used"),
    )

    result = _run_full_step(pg_collection=carrier, losses_reduced=[])

    assert result.loss_dict == {}
```

For a local loss module, supply a plain child with `pp` and `dp_cp`; assert terminal detection and the two-element loss all-reduce use that exact child.

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run pytest -q tests/unit_tests/training/test_train_step_schedule_plumbing.py
```

Expected: old alias forwarding, plain-field access on the union, or second MP reductions fail the new assertions.

- [ ] **Step 3: Implement the branch after optimizer.step**

The schedule call receives the same values:

```python
pg_collection=pg_collection,
p2p_communicator=p2p_communicator,
```

After `optimizer.step()`:

```python
if pg_collection is None:
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

if isinstance(pg_collection, MultiModuleProcessGroupCollection):
    loss_pg_collection = pg_collection.get_loss_module_collection()
    is_last_stage = (
        loss_pg_collection is not None and is_pp_last_stage(loss_pg_collection.pp)
    )
    dp_cp_group = (
        loss_pg_collection.dp_cp if loss_pg_collection is not None else None
    )
else:
    for required in ("mp", "pp", "dp_cp"):
        if getattr(pg_collection, required, None) is None:
            raise ValueError(f"plain pg_collection must define {required}")
    update_successful = logical_and_across_model_parallel_group(
        update_successful, group=pg_collection.mp
    )
    grad_norm = reduce_max_stat_across_model_parallel_group(
        grad_norm, group=pg_collection.mp
    )
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
            num_zeros_in_grad, group=pg_collection.mp
        )
    is_last_stage = is_pp_last_stage(pg_collection.pp)
    dp_cp_group = pg_collection.dp_cp
```

Only the loss-owned terminal branch may use `dp_cp_group`; encoder-only ranks return an empty loss dictionary.

- [ ] **Step 4: Run tests, format, and commit**

```bash
uv run isort megatron/training/training.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
uv run pytest -q tests/unit_tests/training/test_train_step_schedule_plumbing.py
git diff --check
git add megatron/training/training.py \
  tests/unit_tests/training/test_train_step_schedule_plumbing.py
git commit -s -m "Make train step process group carrier generic"
```

### Task 3: Thread the Carrier Through Evaluation

**Files:**
- Modify: `megatron/training/training.py:3245-3505,3890-3930,4035-4280`
- Create: `tests/unit_tests/training/test_evaluation_pg_collection.py`
- Modify: `tests/unit_tests/training/test_pg_collection_plumbing.py`

- [ ] **Step 1: Write failing evaluation identity tests**

Build a one-iteration evaluation harness with mocked timers/model/config. Capture selector and schedule calls:

```python
def test_evaluate_forwards_exact_multimodule_pair(mocker):
    carrier = _multi_carrier(loss_local=False)
    communicator = mock.create_autospec(MultiModulePipelineCommunicator, instance=True)
    schedule = mocker.Mock(return_value=[])
    selector = mocker.patch.object(training_mod, "get_forward_backward_func", return_value=schedule)

    training_mod.evaluate(
        forward_step_func=mock.Mock(),
        data_iterator=iter(()),
        model=[_model_chunk()],
        process_non_loss_data_func=None,
        config=_config(),
        eval_iters=1,
        pg_collection=carrier,
        p2p_communicator=communicator,
    )

    selector.assert_called_once_with(pg_collection=carrier)
    assert schedule.call_args.kwargs["pg_collection"] is carrier
    assert schedule.call_args.kwargs["p2p_communicator"] is communicator
```

Patch MPU terminal/group getters to raise for multi-module tests. Add a loss-rank case proving all-reduce uses the loss child's `dp_cp`.

- [ ] **Step 2: Prove all heterogeneous ranks join non-loss collection**

With `process_non_loss_data_func` set and world-last false, assert the multi-module schedule is still invoked with `collect_non_loss_data=True`, while `collected_non_loss_data` returned to the caller is `None`. With world-last true, retain the returned data.

- [ ] **Step 3: Run tests and verify RED**

```bash
uv run pytest -q tests/unit_tests/training/test_evaluation_pg_collection.py \
  tests/unit_tests/training/test_pg_collection_plumbing.py
```

Expected: evaluation selects MPU schedule, drops both objects, or skips non-reporting ranks.

- [ ] **Step 4: Implement evaluation plumbing**

Add optional `pg_collection` and `p2p_communicator` parameters to `evaluate` and `evaluate_and_print_results`. Resolve the pair before selecting:

```python
p2p_communicator = _resolve_pipeline_communicator(
    pg_collection, p2p_communicator, config
)
forward_backward_func = get_forward_backward_func(pg_collection=pg_collection)
```

Pass both values to every schedule invocation. For loss reduction:

```python
if isinstance(pg_collection, MultiModuleProcessGroupCollection):
    loss_pg_collection = pg_collection.get_loss_module_collection()
    is_loss_terminal = (
        loss_pg_collection is not None
        and is_pp_last_stage(loss_pg_collection.pp, ignore_virtual=True)
    )
    loss_dp_cp_group = (
        loss_pg_collection.dp_cp if loss_pg_collection is not None else None
    )
elif isinstance(pg_collection, ProcessGroupCollection):
    is_loss_terminal = is_pp_last_stage(pg_collection.pp, ignore_virtual=True)
    loss_dp_cp_group = pg_collection.dp_cp
else:
    is_loss_terminal = mpu.is_pipeline_last_stage(ignore_virtual=True)
    loss_dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
```

Use only `loss_dp_cp_group` in the terminal branch. Time-limit coordination remains explicitly world-wide.

For heterogeneous non-loss collection, every rank invokes the schedule. Only `is_last_rank()` retains the returned data.

- [ ] **Step 5: Thread interval/final validation and test callers**

`train` retains the union, removes `model_pg_collection`, resolves its communicator once, selects with `get_forward_backward_func(pg_collection=pg_collection)`, and forwards the exact pair to train-step and interval evaluation. `pretrain` forwards the exact pair to final validation and test.

- [ ] **Step 6: Run tests, format, and commit**

```bash
uv run isort megatron/training/training.py \
  tests/unit_tests/training/test_evaluation_pg_collection.py \
  tests/unit_tests/training/test_pg_collection_plumbing.py
uv run pytest -q tests/unit_tests/training/test_evaluation_pg_collection.py \
  tests/unit_tests/training/test_pg_collection_plumbing.py
git diff --check
git add megatron/training/training.py \
  tests/unit_tests/training/test_evaluation_pg_collection.py \
  tests/unit_tests/training/test_pg_collection_plumbing.py
git commit -s -m "Thread process groups through heterogeneous evaluation"
```

### Task 4: Make Logging Loss-Owner Safe

**Files:**
- Modify: `megatron/training/training.py:2495-2870,3338-3405,3871-3887`
- Create: `tests/unit_tests/training/test_training_log_pg_collection.py`

- [ ] **Step 1: Write failing metric-routing tests**

Add tests proving:

```python
def test_plain_logging_reduces_lr_over_mp(mocker):
    pg_collection = ProcessGroupCollection(mp="mp", dp="dp")
    reduce_stat = mocker.patch.object(
        training_mod, "reduce_max_stat_across_model_parallel_group", return_value=1e-4
    )

    _run_training_log(pg_collection)

    reduce_stat.assert_called_once_with(1e-4, group="mp")


def test_multimodule_logging_does_not_reduce_optimizer_value_again(mocker):
    carrier = _multi_carrier(loss_local=False)
    reduce_stat = mocker.patch.object(
        training_mod,
        "reduce_max_stat_across_model_parallel_group",
        side_effect=AssertionError("second reduction"),
    )

    _run_training_log(carrier)

    reduce_stat.assert_not_called()
```

For encoder-only ranks with MoE/memory flags enabled, patch the tracker and `report_memory` to fail if called. For a local loss child, assert those helpers receive exactly that plain child or its `dp` group.

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run pytest -q tests/unit_tests/training/test_training_log_pg_collection.py
```

Expected: `.mp` or `.dp` access on the union fails, or `None` reaches an MPU-fallback helper.

- [ ] **Step 3: Resolve only the loss-owned metric collection**

At the top of `training_log`:

```python
is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection)
metric_pg_collection = (
    pg_collection.get_loss_module_collection() if is_multimodule else pg_collection
)
```

Plain learning-rate reduction stays unchanged. Multi-module learning rate is already coherent and is not reduced again. Invoke the MoE tracker only when `metric_pg_collection` is not `None`, passing that plain collection. Invoke `report_memory` only when a concrete metric collection exists; never pass `None` from an encoder-only multi-module rank.

In `train`, derive DP world size from the local loss child when present, otherwise from validated `args.data_parallel_size`; do not fall back to MPU for a multi-module carrier.

- [ ] **Step 4: Run tests, format, and commit**

```bash
uv run isort megatron/training/training.py \
  tests/unit_tests/training/test_training_log_pg_collection.py
uv run pytest -q tests/unit_tests/training/test_training_log_pg_collection.py
git diff --check
git add megatron/training/training.py \
  tests/unit_tests/training/test_training_log_pg_collection.py
git commit -s -m "Route heterogeneous training metrics by loss owner"
```

### Task 5: Review and Cog Acceptance

**Files:**
- Verify the files above; no unrelated production edits.

- [ ] **Step 1: Scan the deleted API**

```bash
rg -n 'schedule_pg_collection' megatron examples tests --glob '*.py'
rg -n 'local_collection|get_language_model_collection|has_language_model' \
  megatron examples tests --glob '*.py'
```

Expected: no production or non-absence-test call sites.

- [ ] **Step 2: Run focused tests on Cog**

Dry-run, inspect, then submit:

```bash
cog submit --dry-run \
  --cluster-name cw-dfw-dev \
  --run-name mimo-training-loop-union \
  --gpus 8 --nodes 1 --ntasks-per-node 1 \
  --time 00:30:00 \
  --job-name mimo-training-loop-union \
  --command 'NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=1 uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q tests/unit_tests/training/test_train_step_schedule_plumbing.py tests/unit_tests/training/test_pg_collection_plumbing.py tests/unit_tests/training/test_evaluation_pg_collection.py tests/unit_tests/training/test_training_log_pg_collection.py tests/unit_tests/pipeline_parallel/test_schedules.py tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py'
```

Expected: job completes with exit code zero.

- [ ] **Step 3: Run the checkpoint-disabled MIMO mock train/eval smoke**

Use the existing launcher with save/load disabled and expose `--eval-iters 1 --eval-interval 1` through its existing argument forwarding. This job must complete training plus interval and final evaluation on the non-colocated eight-rank layout. Do not add iterator state or checkpoint work in this plan.

- [ ] **Step 4: Request two-stage review**

The spec reviewer must verify exact carrier identity, no representative PGC, no deleted alias, plain behavior preservation, and encoder-only absence of MPU fallback. The code-quality reviewer must check pairing validation occurs before schedule collectives and that no metric helper interprets encoder-only `None` as MPU fallback.
