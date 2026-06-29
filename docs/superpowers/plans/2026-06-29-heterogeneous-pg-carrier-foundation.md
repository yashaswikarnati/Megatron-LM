# Heterogeneous Process-Group Carrier Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace language-first and first-entry process-group selection with an ordered, module-keyed carrier and make schedule selection consume the generic `pg_collection` argument.

**Architecture:** `MultiModuleProcessGroupCollection` stores global loss ownership and canonical module order while retaining only rank-local module PGCs. `HeteroTopology` owns one distinct grid and PGC per declared module. Schedule selection accepts the same union-typed `pg_collection` name as the schedule itself; the later training-loop plan will remove the remaining outer `schedule_pg_collection` parameter rather than adding a compatibility alias.

**Tech Stack:** Python dataclasses, PyTorch distributed process groups, pytest, Megatron Core pipeline schedules, 8-rank `torch.distributed.run` tests.

---

## File Structure

- `megatron/core/process_groups_config.py`: define the rank-local multi-module carrier and its explicit named resolution API.
- `tests/unit_tests/test_process_groups_config.py`: prove loss ownership, canonical order, validation, and the absence of a representative collection.
- `examples/mimo/training/topology.py`: construct one grid/PGC per module and expose the carrier as `HeteroTopology.pg_collection`.
- `tests/unit_tests/test_mimo_hetero_topology.py`: prove topology preserves module order and distinct per-module PGC identities.
- `megatron/core/pipeline_parallel/schedules.py`: select the bridge schedule from the union `pg_collection` and use generic loss-module terminology during backward.
- `tests/unit_tests/pipeline_parallel/test_schedules.py`: prove plain/none behavior is unchanged and a multi-module carrier selects the bridge schedule.
- `tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py`: migrate distributed schedule fixtures to the new carrier contract.
- `tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py`: migrate MIMO distributed schedule fixtures to the new carrier contract.
- `megatron/training/training.py`: update the existing schedule-selector call to the new keyword without yet changing the outer training API.
- `examples/mimo/pretrain_mimo.py`: read `HeteroTopology.pg_collection`; the separate training-loop plan removes its temporary outer keyword.

This is the first independently testable slice. It deliberately does not change builder, optimizer, evaluation, or checkpoint semantics. Those changes depend on this carrier and are covered by separate implementation plans.

### Task 1: Define the Ordered Rank-Local Carrier

**Files:**
- Modify: `megatron/core/process_groups_config.py:585-716`
- Modify: `tests/unit_tests/test_process_groups_config.py`

- [ ] **Step 1: Write failing carrier behavior tests**

Add these tests to `tests/unit_tests/test_process_groups_config.py`:

```python
def test_multimodule_collection_resolves_loss_module_only_when_local():
    encoder_pgc = ProcessGroupCollection()
    encoder_only = MultiModuleProcessGroupCollection(
        module_pgs={"vision": encoder_pgc},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    assert encoder_only.get_module_collection("vision") is encoder_pgc
    assert encoder_only.get_loss_module_collection() is None

    language_pgc = ProcessGroupCollection()
    language_only = MultiModuleProcessGroupCollection(
        module_pgs={"language": language_pgc},
        loss_module_name="language",
        module_order=("vision", "language"),
    )
    assert language_only.get_loss_module_collection() is language_pgc


def test_multimodule_collection_iteration_uses_global_module_order():
    vision_pgc = ProcessGroupCollection()
    audio_pgc = ProcessGroupCollection()
    collection = MultiModuleProcessGroupCollection(
        module_pgs={"audio": audio_pgc, "vision": vision_pgc},
        loss_module_name="language",
        module_order=("vision", "audio", "language"),
    )

    assert list(collection.keys()) == ["vision", "audio"]
    assert list(collection.values()) == [vision_pgc, audio_pgc]
    assert list(collection.items()) == [("vision", vision_pgc), ("audio", audio_pgc)]
    assert list(collection) == [vision_pgc, audio_pgc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "module_pgs": {"vision": ProcessGroupCollection()},
                "loss_module_name": "language",
                "module_order": ("vision",),
            },
            "loss_module_name",
        ),
        (
            {
                "module_pgs": {"vision": ProcessGroupCollection()},
                "loss_module_name": "language",
                "module_order": ("vision", "vision", "language"),
            },
            "duplicate",
        ),
        (
            {
                "module_pgs": {"audio": ProcessGroupCollection()},
                "loss_module_name": "language",
                "module_order": ("vision", "language"),
            },
            "audio",
        ),
    ],
)
def test_multimodule_collection_validates_global_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultiModuleProcessGroupCollection(**kwargs)


def test_multimodule_collection_has_no_representative_collection():
    collection = MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection()},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    assert not hasattr(collection, "local_collection")
    assert not hasattr(collection, "get_language_model_collection")
```

- [ ] **Step 2: Run the carrier tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit_tests/test_process_groups_config.py \
  -k 'multimodule_collection'
```

Expected: collection construction fails because `loss_module_name` and `module_order` are not accepted, and the representative API still exists.

- [ ] **Step 3: Replace the carrier implementation**

Replace `MultiModuleProcessGroupCollection` with this API, retaining its concise `__repr__` style:

```python
@dataclass
class MultiModuleProcessGroupCollection:
    """Rank-local process groups for a globally ordered multi-module pipeline."""

    module_pgs: Dict[str, ProcessGroupCollection]
    loss_module_name: str
    module_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.module_pgs:
            raise ValueError("module_pgs cannot be empty")
        if len(set(self.module_order)) != len(self.module_order):
            raise ValueError(f"module_order contains duplicate names: {self.module_order}")
        if self.loss_module_name not in self.module_order:
            raise ValueError(
                f"loss_module_name {self.loss_module_name!r} is absent from "
                f"module_order {self.module_order}"
            )
        unknown_local_modules = set(self.module_pgs).difference(self.module_order)
        if unknown_local_modules:
            raise ValueError(
                f"Local modules are absent from module_order: {sorted(unknown_local_modules)}"
            )

    def get_module_collection(self, module_name: str) -> ProcessGroupCollection:
        if module_name not in self.module_pgs:
            raise ValueError(
                f"Module {module_name!r} is not active on this rank; "
                f"active modules: {list(self.keys())}"
            )
        return self.module_pgs[module_name]

    def get_loss_module_collection(self) -> Optional[ProcessGroupCollection]:
        return self.module_pgs.get(self.loss_module_name)

    def __len__(self) -> int:
        return len(self.module_pgs)

    def __getitem__(self, module_name: str) -> ProcessGroupCollection:
        return self.module_pgs[module_name]

    def __iter__(self):
        return iter(self.values())

    def keys(self):
        return (name for name in self.module_order if name in self.module_pgs)

    def values(self):
        return (self.module_pgs[name] for name in self.keys())

    def items(self):
        return ((name, self.module_pgs[name]) for name in self.keys())
```

Delete `language_model_module_name`, `get_language_model_collection()`, `get_language_model_cp_size()`, `has_language_model()`, and `local_collection`. Do not add deprecated aliases.

- [ ] **Step 4: Run the carrier tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/unit_tests/test_process_groups_config.py \
  -k 'multimodule_collection'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the carrier API**

```bash
git add megatron/core/process_groups_config.py tests/unit_tests/test_process_groups_config.py
git commit -s -m "Refine multi-module process group ownership"
```

### Task 2: Make Topology Ownership Module-Ordered

**Files:**
- Modify: `examples/mimo/training/topology.py:63-116,230-249`
- Modify: `examples/mimo/pretrain_mimo.py:145`
- Modify: `tests/unit_tests/test_mimo_hetero_topology.py`

- [ ] **Step 1: Write failing topology tests**

Add to `tests/unit_tests/test_mimo_hetero_topology.py`:

```python
def test_topology_exposes_ordered_pg_collection():
    specs = _specs()
    topology = create_topology(specs)
    try:
        assert topology.pg_collection.module_order == tuple(spec.name for spec in specs)
        assert topology.pg_collection.loss_module_name == MIMO_LANGUAGE_MODULE_KEY
        assert list(topology.pg_collection.keys()) == [
            spec.name for spec in specs if topology.grids[spec.name].is_current_rank_in_grid()
        ]
    finally:
        topology.destroy()


def test_topology_keeps_distinct_collections_for_identical_module_specs(mocker):
    grids = {"vision": mocker.Mock(), "audio": mocker.Mock()}
    module_pgs = {
        "vision": ProcessGroupCollection(),
        "audio": ProcessGroupCollection(),
    }
    for grid in grids.values():
        grid.is_current_rank_in_grid.return_value = True

    collection = build_multi_module_pg_collection(
        grids,
        module_pgs,
        loss_module_name="language",
        module_order=("vision", "audio", "language"),
    )

    assert collection["vision"] is module_pgs["vision"]
    assert collection["audio"] is module_pgs["audio"]
    assert collection["vision"] is not collection["audio"]
```

Use the file's existing `mocker` fixture/import conventions rather than introducing a new mocking dependency.

- [ ] **Step 2: Run the topology tests and verify RED**

Run:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/test_mimo_hetero_topology.py
```

Expected: tests fail because `HeteroTopology.pg_collection` and `build_multi_module_pg_collection` do not exist.

- [ ] **Step 3: Rename and populate the topology carrier**

Use these signatures and construction rules:

```python
@dataclass
class HeteroTopology:
    grids: dict[str, HyperCommGrid]
    module_pgs: dict[str, ProcessGroupCollection]
    pg_collection: MultiModuleProcessGroupCollection


def build_multi_module_pg_collection(
    grids: dict[str, HyperCommGrid],
    module_pgs: dict[str, ProcessGroupCollection],
    loss_module_name: str,
    module_order: tuple[str, ...],
) -> MultiModuleProcessGroupCollection:
    rank_modules = {
        name: module_pgs[name]
        for name in module_order
        if grids[name].is_current_rank_in_grid()
    }
    return MultiModuleProcessGroupCollection(
        module_pgs=rank_modules,
        loss_module_name=loss_module_name,
        module_order=module_order,
    )
```

In `create_topology`, derive `module_order = tuple(spec.name for spec in specs)`, build every grid and PGC independently in that order, and return:

```python
pg_collection = build_multi_module_pg_collection(
    grids,
    module_pgs,
    loss_module_name=MIMO_LANGUAGE_MODULE_KEY,
    module_order=module_order,
)
return HeteroTopology(grids=grids, module_pgs=module_pgs, pg_collection=pg_collection)
```

For exception cleanup, make `HeteroTopology.pg_collection` optional or destroy the grids/PGCs directly; never synthesize an invalid empty carrier. In `examples/mimo/pretrain_mimo.py`, temporarily pass `topology.pg_collection` through the existing outer keyword. The next plan removes that keyword everywhere.

- [ ] **Step 4: Run the topology tests and verify GREEN**

Run:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/test_mimo_hetero_topology.py
```

Expected: all topology tests pass on eight ranks.

- [ ] **Step 5: Format imports and commit topology ownership**

```bash
uv run isort examples/mimo/training/topology.py examples/mimo/pretrain_mimo.py \
  tests/unit_tests/test_mimo_hetero_topology.py
git add examples/mimo/training/topology.py examples/mimo/pretrain_mimo.py \
  tests/unit_tests/test_mimo_hetero_topology.py
git commit -s -m "Make MIMO topology process groups module ordered"
```

### Task 3: Use `pg_collection` for Schedule Selection

**Files:**
- Modify: `megatron/core/pipeline_parallel/schedules.py:48-151,595-625,2280-2290`
- Modify: `megatron/training/training.py` at the existing `get_forward_backward_func` call
- Modify: `tests/unit_tests/pipeline_parallel/test_schedules.py:47-77`
- Modify: `tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py:408-423`
- Modify: `tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py:669-681`

- [ ] **Step 1: Write failing schedule-selection tests**

Extend `test_get_forward_backward_func` or add adjacent tests in `test_schedules.py`:

```python
def test_get_forward_backward_func_selects_multimodule_schedule_from_pg_collection():
    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection()},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    assert (
        schedule.get_forward_backward_func(pg_collection=pg_collection)
        is schedule.forward_backward_pipelining_without_interleaving
    )


def test_get_forward_backward_func_has_no_schedule_pg_collection_alias():
    pg_collection = MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection()},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    with pytest.raises(TypeError, match="schedule_pg_collection"):
        schedule.get_forward_backward_func(schedule_pg_collection=pg_collection)
```

Import `MultiModuleProcessGroupCollection` beside `ProcessGroupCollection`.

- [ ] **Step 2: Run the schedule selector tests and verify RED**

Run:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  -k 'get_forward_backward_func'
```

Expected: the new keyword is rejected and the old keyword is still accepted.

- [ ] **Step 3: Rename the schedule-selector API and loss policy**

Change the selector signature and first branch to:

```python
def get_forward_backward_func(
    pp_size: Optional[int] = None,
    vp_size: Optional[int] = None,
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
):
    if isinstance(pg_collection, MultiModuleProcessGroupCollection):
        return forward_backward_pipelining_without_interleaving
```

Rename `backward_step_multimodule(..., language_model_module_name)` to `loss_module_name`, and associate a scalar terminal loss with that key:

```python
if not isinstance(output_tensor, dict):
    output_tensor = {loss_module_name: output_tensor}
```

Bind it in the bridge schedule with:

```python
backward_func = partial(
    backward_step_multimodule,
    loss_module_name=pg_collection.loss_module_name,
)
```

Update the existing `training.py` selector call to `get_forward_backward_func(pg_collection=...)`; this task does not retain an alias in `schedules.py`.

- [ ] **Step 4: Migrate distributed fixtures**

Every fixture must pass global policy even when the loss module is not local:

```python
pg_collection = MultiModuleProcessGroupCollection(
    module_pgs=module_pgs,
    loss_module_name=MIMO_LANGUAGE_MODULE_KEY,
    module_order=tuple(module_to_grid_map),
)
```

For generic multimodule schedule tests, use their existing module list and explicit loss key rather than importing the MIMO language constant.

- [ ] **Step 5: Run schedule tests and verify GREEN**

Run:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  -k 'get_forward_backward_func'
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Format imports and commit schedule selection**

```bash
uv run isort megatron/core/pipeline_parallel/schedules.py megatron/training/training.py \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py \
  tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py
git add megatron/core/pipeline_parallel/schedules.py megatron/training/training.py \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py \
  tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py
git commit -s -m "Select heterogeneous schedules from pg collection"
```

### Task 4: Prove the Foundation Has No Representative Selection

**Files:**
- Verify only; modify the files above only if a focused test exposes a defect.

- [ ] **Step 1: Scan obsolete carrier symbols**

Run:

```bash
rg -n 'local_collection|language_model_module_name|get_language_model_collection|has_language_model' \
  megatron/core/process_groups_config.py megatron/core/pipeline_parallel/schedules.py \
  examples/mimo/training/topology.py tests/unit_tests --glob '*.py'
```

Expected: no obsolete carrier symbol remains in the migrated files. References in later, not-yet-migrated orchestration files are recorded for the next plan rather than hidden behind aliases.

- [ ] **Step 2: Scan the schedule selector alias**

Run:

```bash
rg -n 'get_forward_backward_func\([^)]*schedule_pg_collection|schedule_pg_collection=' \
  megatron/core/pipeline_parallel tests/unit_tests/pipeline_parallel \
  tests/unit_tests/models/mimo --glob '*.py'
```

Expected: no match.

- [ ] **Step 3: Run the complete focused foundation suite**

Run:

```bash
uv run pytest -q tests/unit_tests/test_process_groups_config.py
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/test_mimo_hetero_topology.py \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  tests/unit_tests/pipeline_parallel/test_multimodule_schedules.py \
  tests/unit_tests/models/mimo/test_mimo_1f1b_schedule.py
git diff --check
```

Expected: all tests pass and `git diff --check` prints nothing.

- [ ] **Step 4: Request two-stage review**

Dispatch a spec-compliance reviewer against the approved design and the commits from Tasks 1-3. After any fixes and re-review, dispatch a separate code-quality/adversarial reviewer. Required review questions:

```text
1. Can any generic call site still select the language module or first mapping entry as a representative PGC?
2. Is loss_module_name valid as global metadata when absent from module_pgs on this rank?
3. Does every iteration over local modules follow module_order rather than dict insertion order?
4. Did the patch accidentally alias PGCs for modules with identical layouts?
5. Did ordinary plain/None schedule selection change?
6. Is any compatibility alias retained for schedule_pg_collection in get_forward_backward_func?
```

- [ ] **Step 5: Commit review fixes, if any**

```bash
git add megatron/core/process_groups_config.py megatron/core/pipeline_parallel/schedules.py \
  megatron/training/training.py examples/mimo/training/topology.py \
  examples/mimo/pretrain_mimo.py tests/unit_tests
git commit -s -m "Address process group carrier review"
```

Skip this commit when review produces no code changes.
