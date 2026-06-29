# Generic Heterogeneous Training Process-Group Design

**Status:** Proposed for implementation on PR #43 at `5408946a31`, stacked on PR #5516.

## Objective

Make the Megatron training loop support both ordinary single-grid models and heterogeneous
multi-module models through one process-group argument, without choosing an arbitrary module as
the representative of a multi-module topology.

Non-colocated MIMO is the required configuration. The union API must represent colocated module
inventories without choosing by dictionary order, while the new bridge entry point rejects
colocated execution during startup until its initialization, scheduling, and checkpoint semantics
are designed together.

For the supported non-colocated, same-topology MCore DDP/distributed-optimizer path, checkpointing
targets parity for the enumerated `torch_dist` modes below. Heterogeneous training must not disable
async save, fully-parallel load, data-parallel random initialization, or non-persistent recovery
merely because MPU globals are unavailable. Topology resharding remains a separate feature.

## Current Problems

PR #43 currently carries two related values through the training loop:

- `pg_collection`, a plain `ProcessGroupCollection` used by legacy training helpers; and
- `schedule_pg_collection`, a `MultiModuleProcessGroupCollection` used by the heterogeneous
  pipeline schedule.

Commit `60aa6f198f` partially removes the duplication by adding `local_collection`, but that
property is not a valid generic abstraction:

- it requires exactly one module even though `MultiModuleProcessGroupCollection` documents
  colocated and multi-encoder ranks;
- it is dereferenced early in `pretrain()` and `train()`, discarding the multi-module carrier;
- it uses `assert` followed by `next(iter(...))`, so optimized Python removes the guard and turns
  an ambiguous colocated collection into an insertion-order fallback; and
- it leaves setup, train-step, logging, evaluation, and checkpointing dependent on a second plain
  collection.

There are two additional correctness gaps that the process-group refactor must expose rather than
hide:

1. Heterogeneous evaluation does not receive the multi-module collection or communicator and
   selects its schedule from MPU globals. `eval_iters=0` only masks this path.
2. `MimoOptimizer` reaches world overflow consensus only after each inner FP16 optimizer has
   updated its module-local loss scaler. If one grid overflows, all ranks skip the step but the
   encoder and language loss scalers can diverge.

## Design Principles

1. **One orchestration argument.** `pretrain`, model setup, `train`, `train_step`, evaluation, and
   schedule selection receive one value:

   ```python
   ProcessGroupCollection | MultiModuleProcessGroupCollection
   ```

2. **The multi-module collection is rank-local.** It contains every module collection active on
   the current rank. A non-colocated rank has one entry; a colocated rank may have several.

3. **Resolution is purpose-specific.** A caller asks for a named module, the loss-owning module,
   or a strictly single local module. There is no language-first, global-rank-first, or dictionary-
   order fallback.

4. **The union stays at orchestration boundaries.** DDP, an individual module optimizer, ordinary
   checkpoint primitives, and low-level collectives continue to receive a plain collection or an
   explicit process group.

5. **The optimizer owns optimizer statistics.** A heterogeneous optimizer returns coherent step
   success, loss-scale state, gradient norm, and zero count across its coordination group. The
   training loop must not rediscover heterogeneous optimizer topology.

6. **Single-grid behavior remains unchanged.** Ordinary Megatron continues to materialize a plain
   collection from MPU and uses the existing schedules, reductions, logging, and checkpoint APIs.

7. **Unsupported ambiguity fails before collectives.** Runtime validation uses `ValueError` or
   `NotImplementedError`, never `assert`, for topology-dependent behavior.

## Process-Group Model

`MultiModuleProcessGroupCollection` remains the mapping from module name to the rank-local plain
collections. Its generic policy field identifies the module that owns the scalar training loss;
MIMO binds that field to its language module. The loss-module name is identical on every rank, but
the corresponding plain collection is present only on ranks that participate in that module.
Because this API is introduced within the unmerged stack, the language-named field and accessors
are replaced rather than retained as aliases. MIMO-specific code supplies its language module name
as the generic `loss_module_name`.

The collection exposes only explicit operations:

- `get_module_collection(module_name)` for module-specific work;
- `get_loss_module_collection()` for loss and terminal-stage work on participating ranks; and
- `require_single_module()` returning `(module_name, collection)` only when exactly one module is
  active on this rank.

`get_loss_module_collection()` returns `None` on ranks outside the loss module.
`require_single_module()` raises `ValueError` when the rank is colocated. The method name makes the
restriction visible at every call site and preserves the module name for checkpoint namespaces.
The `local_collection` property is removed.

Global batch accounting does not use the optional rank-local loss collection. The MIMO entry point
passes the language-grid DP size into `validate_args()` as the `data_parallel_size` override, so
global/evaluation batch defaults and divisibility checks are derived from the correct scalar. It
then validates that value against the loss-grid specification before global variables and the
microbatch calculator are initialized. The post-validation assignment currently in
`pretrain_mimo.py` is removed because it occurs after batch-derived fields have already been set.

## Training-Loop Plumbing

### Pretrain and Initialization

The public argument is named `pg_collection` and accepts the union. Because
`schedule_pg_collection` is already present on `origin/main`, `pretrain()`, `train()`, and
`train_step()` retain it as a deprecated boundary alias. A shared normalizer immediately converts
the alias to `pg_collection` and raises `ValueError` when both names receive different objects.
Internal callers and local variables use only `pg_collection`.

When no collection is supplied, ordinary initialization runs and then creates a plain collection
from MPU as it does today.

When a multi-module collection is supplied:

- the original union object remains the training-loop carrier;
- non-colocated bootstrap code may call `require_single_module()` for seed/JIT inputs that truly
  require one collection; and
- colocated initialization does not select a representative module. Per-module RNG seeding occurs
  in the multi-module builder. Any generic initialization operation that cannot work without one
  collection must fail clearly or be decomposed before colocated support is claimed.

The initial heterogeneous startup validator requires all of the following before model
construction:

- module grids are disjoint and partition `torch.distributed.group.WORLD`;
- every rank has exactly one local module;
- one loss module is named consistently on every rank;
- the loss module contains the global last rank used by the current writer/logging stack; and
- the carrier is paired with a `MultiModulePipelineCommunicator`.

The global-last restriction is an explicit initial compatibility boundary, not a generic
requirement of the union. Removing it later requires routing writers and reduced metrics to an
arbitrary topology-provided reporting rank.

### Model Builders

The `ModelBuilder` interface accepts the union because model construction is an orchestration
boundary. Ordinary GPT/hybrid builders validate and consume a plain collection. `MimoModelBuilder`
validates and consumes a multi-module collection.

For MIMO, the passed rank-local mapping is authoritative for active module process groups. The
builder, DDP wrappers, gradient finalizer, and `MimoOptimizer` use those same plain collection
instances. The topology collection constructor is expanded to include optimizer-required groups,
including `intra_dist_opt`, so `MimoOptimizer` does not reconstruct a second collection from
`HyperCommGrid`. `MimoModel.pg_collection` stores the complete rank-local multi-module collection,
never a language-only plain collection, and optimizer dispatch reads that collection from the
model.

`HeteroTopology` remains responsible for grid placement, group creation/destruction, and
cross-module communication metadata. It is not a second source for selecting the current rank's
active process groups. The builder validates that the keys and object identities in the passed
rank-local mapping match the active subset constructed by `HeteroTopology`.

The builder cleanup also removes the full `argparse.Namespace` from `MimoBuildConfig`. The build
config contains explicit typed build values. A temporary non-serialized topology handle may remain
until the ModelBuilder runtime-context API is designed, but it cannot duplicate the authoritative
rank-local process-group mapping.

Unused builder parameters are handled honestly: parameters required by the abstract interface are
validated or documented as unsupported by MIMO; they are not silently accepted when enabling them
would produce incorrect behavior.

### Training and Evaluation

`train()` and `train_step()` receive one union value and pass that same object to:

- schedule selection;
- the selected forward/backward schedule;
- loss/logging helpers; and
- checkpoint adapters.

The separate `model_pg_collection` and `schedule_pg_collection` variables are removed.

`evaluate()` and `evaluate_and_print_results()` receive the same collection and P2P communicator
and select/invoke the schedule using the same rules as training. A heterogeneous run with
`eval_iters > 0` must therefore exercise the multi-module schedule rather than MPU globals.

Schedule selection passes `p2p_communicator` to `get_forward_backward_func()` and selects the
bridge schedule when that object is a `MultiModulePipelineCommunicator`. The bridge schedule then
requires the multi-module collection. This preserves the ordinary colocated schedule for existing
colocated MIMO paths that use a plain collection; no second layout selector is introduced.

The valid pairs are checked before schedule selection:

- plain collection with an ordinary `P2PCommunicator`, or with no explicit communicator on the MPU
  fallback path; and
- multi-module collection with a `MultiModulePipelineCommunicator`.

A mixed pair raises `ValueError` before any schedule collective.

Both evaluation schedule invocations receive the same carrier and communicator as training. When
`collect_non_loss_data` is requested, every heterogeneous rank participates in the schedule and
only the validated logging owner processes the returned data. Evaluation terminal-stage and DP
loss operations use the optional loss-module collection. Full-validation and time-limit control
use the explicit world coordination group rather than MPU. The initial requirement that the loss
module contains global last rank keeps existing writer and `print_rank_last` behavior valid.

### Losses, Batch Size, and Logging

Loss reduction and terminal-stage detection call `get_loss_module_collection()` explicitly.
Encoder-only ranks receive `None` and do not fabricate a plain collection for those operations.
Global batch and scheduler increments continue using the validated process-wide
`args.data_parallel_size` scalar.

Ordinary optimizer statistics retain their existing model-parallel reduction when the carrier is
a plain collection. The multi-module path requires its coordinating optimizer to return globally
coherent values and performs no second train-loop reduction. This follows from the generic carrier
type and does not add an optimizer capability registry or an `isinstance(MimoOptimizer)` branch.

Logging code resolves a plain collection only for the metric whose semantics require it. It does
not accept a preselected generic "primary" collection. MIMO step statistics are logged as returned
by the optimizer. Loss metrics use the loss-module collection. Metrics that have no defined
heterogeneous meaning must be disabled with an explicit error rather than reduced on an arbitrary
module.

## MIMO Optimizer Coordination

`MimoOptimizer.step()` returns final heterogeneous step statistics. For the initial non-colocated
implementation, `HeteroTopology` validates that the module grids partition the entire distributed
world. `MimoModelBuilder` passes `torch.distributed.group.WORLD` explicitly to `MimoModel`, which
passes it to `MimoOptimizer`. No duplicate NCCL group is created. Supporting a subset of world is
deferred until a concrete layout requires it.

The required sequence is:

1. Each active inner optimizer copies/unscales gradients and detects module-local overflow by
   calling `prepare_grads(update_grad_scaler=False)`.
2. MIMO reduces overflow across the coordination group.
3. Every active inner optimizer updates its loss scaler exactly once using the same global overflow
   result.
4. If overflow occurred, every rank returns the same unsuccessful result before norm collectives.
5. Per-module gradient norms are placed in a stable module-keyed vector, reduced across the
   coordination group, and combined.
6. Global clipping preserves the current MIMO precision-aware gradient handling.
7. Inner optimizers step. `step_with_ready_grads()` is required to return `True` after the global
   overflow decision; a false return is an invariant violation, not a recoverable post-mutation
   failure.
8. When requested, per-module zero counts are placed in a stable module-keyed vector, reduced, and
   summed so every rank returns the same value.

The optimizer API change is explicit and backward-compatible:

```python
def prepare_grads(self, *, update_grad_scaler: bool = True) -> bool: ...
def update_grad_scaler(self, found_inf: bool) -> None: ...
```

`MixedPrecisionOptimizer.prepare_grads()` retains its current copy/unscale/check behavior and calls
`update_grad_scaler()` only when requested. `update_grad_scaler()` updates the complete
`DynamicGradScaler` state: scale, growth tracker, and hysteresis tracker. `FP32Optimizer` implements
a no-op. `ChainedOptimizer` forwards both operations to every inner optimizer. Existing callers
omit the keyword and retain current behavior.

Each module optimizer continues to checkpoint its own scaler state. New heterogeneous checkpoints
contain identical scaler states because every active optimizer receives the same global overflow
decision. After load, MIMO compares the complete scaler state across the coordination group before
the first step. A checkpoint with divergent per-module scaler state is rejected with an explicit
compatibility error rather than silently choosing one module's scale. In the initial FP16 path,
every rank must have at least one active inner optimizer; a rank whose sole module is frozen or
stubbed is rejected during startup because it has no canonical local scaler. `get_loss_scale()`
returns the locally active scaler only after this equality invariant has been established.

The initial MIMO optimizer preserves its current single global norm and precision-aware clipping.
Configurations using `grad_norm_skip_threshold` or registered separate grad-norm groups are
rejected during startup; coordinating those decisions across module grids requires a separate
optimizer design. This process-group change does not duplicate `ChainedOptimizer` private clipping
logic.

## Checkpointing

The training-loop checkpoint helpers receive the union. The public low-level `save_checkpoint()`
retains explicit TP/PP/DP/DP-CP/expert-DP group arguments. `load_checkpoint()` adds the missing
expert-DP argument so save and load can select the same fully-parallel group. A small training-layer
adapter resolves the union at that boundary.

The synchronous save finalizer also stops reading TP/PP rank and world size from MPU. When explicit
groups are supplied, it derives both rank and size with `torch.distributed.get_rank(group)` and
`torch.distributed.get_world_size(group)`; the MPU fallback remains only for ordinary callers that
do not supply groups.

The async callback captures the resolved integer TP/PP ranks and sizes before the request is
scheduled. It does not retain a process-group object for later semantic discovery, and async queue
finalization remains world-coordinated because the current module grids partition world.

- Plain collection: preserve existing behavior.
- Non-colocated multi-module collection: call `require_single_module()`, use that module's groups,
  and preserve an explicit module RNG namespace.
- Colocated multi-module collection: do not select the loss module or first module. The current
  checkpoint adapter has one group set and one RNG namespace, so this combination is rejected by
  startup feature validation before model construction when save or load is configured.

The RNG namespace is model checkpoint metadata, not process-group policy. `MimoModel` declares its
`rng_state_key_prefix` explicitly from the active module during construction. The generic adapter
retrieves that attribute from the wrapped model and requires a non-empty prefix for a multi-module
carrier; generic training code does not construct a MIMO-specific prefix.

The supported parity matrix is:

| Heterogeneous option | Status |
| --- | --- |
| Persistent synchronous `torch_dist` checkpoint | Supported |
| Persistent async `torch_dist` checkpoint and persistent worker | Supported |
| Fully-parallel save and load over DP | Supported |
| Fully-parallel save and load over expert-DP | Supported |
| Stock fully-parallel load exchange algorithms | Preserve stock validation/warnings |
| `data_parallel_random_init` seed, parameter broadcast, and RNG restore | Supported |
| Global non-persistent `torch_dist` checkpoint | Supported |
| Local non-persistent checkpoint, including async | Supported when the stock NVRx dependency is available |
| Local checkpoint replication | Supported when configured by stock local-checkpoint arguments |
| `ckpt_assume_constant_structure` and cached strategy reuse | Supported |
| Distributed optimizer, scheduler, scaler, RNG, iteration, and consumed-sample restore | Supported |
| Legacy checkpoint format | Rejected at startup |
| Megatron FSDP or Torch FSDP2 | Rejected at startup |
| RL reference reload or MoE upcycling | Rejected at startup |
| Checkpoint format conversion | Rejected at startup |

The load API gains an explicit `expt_dp_group` and threads both `dp_cp_group` and `expt_dp_group`
through `load_checkpoint()`, `_load_base_checkpoint()`, global load, and non-persistent load.
Fully-parallel load selects only those supplied groups. Its ShardedObject exchange uses the same
selected group for group size and `all_gather_object`; it never expands to world implicitly.

Local non-persistent tensor-aware save and restore receive the resolved DP-CP group instead of
calling MPU. Global non-persistent finalization uses its selected `save_dir` for tracker retention
and cleanup.

The common checkpoint metadata stores a module-keyed topology map containing each module's name,
rank span, TP, PP, CP, DP, EP, and expert-TP dimensions. Before generating or loading sharded model,
optimizer, or RNG state, resume validates the runtime topology against that map. MIMO
uses its local module's checkpoint dimensions for RNG and optimizer compatibility decisions instead
of the single language-oriented TP/PP/DP values in common `args`. The initial parity target requires
an identical module topology on resume; heterogeneous topology resharding remains out of scope and
fails before checkpoint collectives.

The same adapter is used by initial load, interval save, signal save, duration/iteration exit save,
and final save. Direct low-level checkpoint calls in RL reload, MoE upcycling, and conversion either
route through the adapter when those model modes become supported or reject the heterogeneous
carrier explicitly.

The mock external dataloader is stateless; saving or restoring its iterator state is outside this
PR. Resume correctness uses restored consumed-sample accounting and training state, not a new mock
dataloader checkpoint format.

### Data-Parallel Random Initialization

Data-parallel random initialization is part of checkpoint parity, not an unsupported checkpoint
mode. Per-module seeding uses the module's plain `dp` group, matching ordinary Megatron semantics.
After each active module is DDP-wrapped, `MimoModelBuilder` calls `broadcast_params()` when
`data_parallel_random_init=True`; each DDP wrapper already owns the correct module-local dense and
expert groups. RNG save gathers over that same module DP group and load indexes by its local DP
rank and module namespace. The initial CP=1 topology restriction remains explicit because the
current RNG checkpoint representation does not encode independent CP states.

## Colocated Support Boundary

| Layout | Carrier and schedule | Status |
| --- | --- | --- |
| Legacy colocated MIMO | Plain collection, ordinary schedule | Unchanged |
| Non-colocated heterogeneous MIMO | Multi-module collection and bridge communicator | Supported |
| Colocated bridge MIMO | Multi-module collection and bridge communicator | Rejected during pretrain startup validation |

The orchestration API and collection can represent colocated ranks without order-dependent
selection, but the new bridge entry point does not claim colocated execution. If save/load is
configured for an unsupported layout, validation fails before model construction rather than at
the first checkpoint interval.

Partial grid overlap, arbitrary multi-encoder fan-in policy, and independent per-module learning
rates are outside this change. The API must not prevent those extensions.

Generic training modules may import only the generic multi-module collection and communicator
interfaces. MIMO module-name constants, builders, optimizer classes, and topology policy remain in
MIMO-specific code.

## Review-Comment Scope

The implementation also addresses the unresolved PR comments that intersect this design:

- replace the language/first-module and `local_collection` selection;
- stop attaching a language-only plain collection to `MimoModel`;
- make the MIMO builder consume its process-group argument;
- remove raw parsed args from `MimoBuildConfig`;
- make RNG checkpoint namespace usage explicit;
- remove or validate unused builder options;
- preserve concise docstrings and avoid unrelated optimizer-docstring edits; and
- keep mock valid/test iterators and CLI-controlled eval/rerun behavior.

No GitHub review replies or thread resolutions are part of this implementation unless explicitly
requested.

## Implementation Milestones

The work is split into five independently testable slices:

1. Union plumbing, explicit resolution, batch accounting, schedule selection, and evaluation.
2. MIMO builder/process-group source-of-truth cleanup and typed build configuration.
3. Optimizer overflow/scaler/statistics coordination.
4. Persistent `torch_dist` parity: sync/async lifecycle, fully-parallel DP/expert-DP save/load,
   DP-random initialization, and non-colocated resume.
5. Global and local non-persistent `torch_dist` parity with the same explicit-group adapter.

Each milestone leaves ordinary Megatron and the supported MIMO path runnable. Detailed red/green
commands, review records, and Cog submissions belong to the implementation plans for these slices.

## Verification Strategy

Focused tests cover:

- explicit named/loss/single-module resolution, including reversed insertion order and optimized
  Python behavior;
- one union argument through pretrain, setup, train, train-step, evaluation, and checkpoint calls;
- ordinary single-collection fallback unchanged;
- non-colocated rank-local topology with encoder ranks 0-3 and language ranks 4-7;
- colocated ambiguity either handled by an explicit operation or rejected before collectives;
- MIMO overflow on only one grid with identical scaler state and skipped step on every rank;
- complete scaler-state equality after save/resume, including growth and hysteresis trackers;
- startup rejection for an FP16 rank with no active inner optimizer;
- globally identical success, gradient norm, and zero count;
- cross-grid evaluation with `eval_iters > 0`; and
- stock checkpoint save/load plus non-colocated MIMO save/resume through the training loop;
- repeated async saves with tracker advancement only after finalization;
- fully-parallel save/load over both DP and nontrivial expert-DP groups without MPU fallback;
- data-parallel random initialization with module-local broadcast and exact RNG restoration;
- global and local non-persistent save/resume.

The FP16 coordination proof uses a real 8-rank test in
`tests/unit_tests/models/mimo/test_mimo_optimizer_coordination.py`; the BF16 launcher is not
accepted as evidence for scaler correctness. An evaluation run is accepted only when its log
contains the validation-loss record for the expected iteration. A resume run is accepted only when
it loads iteration 2, runs iteration 3, and advances the checkpoint tracker to 3.

Evaluation unit coverage captures both normal and `collect_non_loss_data` schedule calls and proves
that every heterogeneous rank receives the same carrier/communicator pair. Startup tests cover
invalid plain/multi communicator pairs, loss-grid placement, unsupported checkpoint formats, and
colocated bridge input. One parameterized startup-validation test has a case for every rejected
checkpoint/optimizer option in the support matrices: legacy format, Megatron FSDP, Torch FSDP2,
RL reload, MoE upcycling, checkpoint conversion, `grad_norm_skip_threshold`, registered separate
grad-norm groups, and an FP16 rank with no active inner optimizer. Each case proves rejection occurs
before model construction or any collective.

Checkpoint acceptance compares checkpointed and freshly loaded state at iteration 2: every model
and optimizer shard, scheduler/scaler state, RNG state, iteration, and consumed-sample count must
match. The resumed job must then complete iteration 3, but its loss is not compared with an
uninterrupted run because the intentionally stateless mock iterator restarts its private generator.
The async tracker remains on the previous completed checkpoint until finalization and advances only
after the async queue completes.

The save-versus-load state oracle applies to persistent, global non-persistent, and each local
checkpoint algorithm. It compares every module/rank model and optimizer shard, scheduler, scaler,
RNG, iteration, and consumed samples immediately after load. Integer and serialized state use exact
equality; floating-point tensors use the same dtype-aware tolerances as existing
distributed-checkpoint round-trip tests.

The primary combined case enables async persistent-worker save, fully-parallel DP save/load,
`data_parallel_random_init`, and cached checkpoint structure together. A separate expert-DP case
uses language-module expert-DP larger than one, for example EP=2 on the four language ranks, and
asserts exact group identity and membership. Focused tests exercise all stock fully-parallel
exchange algorithms and prove ShardedObject exchange stays inside the selected group.

Cached-structure coverage performs at least two completed async saves in one process with the same
checkpointing context, observes both tracker transitions, and proves cached plan/metadata reuse. A
failed or aborted async request must not advance the tracker.

Global non-persistent tests cover newer, equal, and older precedence relative to the persistent
checkpoint and perform retention only after successful finalization. Local recovery covers
`atomic` and `fully_parallel`, sync and async, reconstruction with a fresh manager context, and
configured replication. Local recovery resumes within the same Slurm allocation and node mapping,
matching the stock local-checkpoint contract.

Replication coverage removes at least one primary shard in every relevant replication clique,
reconstructs a fresh manager, restores from a peer replica, and passes the same save-versus-load
oracle. Missing more primary shards than the configured replication factor can recover must fail
cleanly rather than resume from incomplete state.

Topology-metadata coverage proves the complete module map is identical on every rank. Starting from
a saved topology, parameterized cases change the module set, rank span, TP, PP, CP, DP, EP, and
expert-TP fields independently and verify deterministic rejection before any checkpoint collective.

A parameterized routing test covers interval, signal, duration exit, iteration exit, and final-save
triggers so every claimed training-loop checkpoint entry point is proven to use the same adapter.

Every TDD increment records the exact selector, RED baseline SHA, expected failure, RED exit status,
and GREEN result. Every review record names the reviewed commit, spec reviewer, adversarial
reviewer, findings, and resolution.

Cluster validation uses CW DFW through `cog`:

1. run `cog doctor --cluster-name cw-dfw-dev`;
2. run each fully specified submission with `cog submit ... --dry-run` and inspect workspace sync,
   image, environment, one-task-per-node layout, and Slurm command;
3. run focused 8-GPU distributed topology, runtime, schedule, optimizer, and checkpoint tests with
   `NCCL_MAX_NCHANNELS=1`, `NCCL_NVLS_ENABLE=0`, and `CUDA_DEVICE_MAX_CONNECTIONS=1`;
4. run the heterogeneous mock training launcher with evaluation enabled; and
5. run save-at-iteration-2/resume-to-iteration-3 checkpoint parity jobs for DP and expert-DP; and
6. run at least one multi-node async/fully-parallel resume job using one Slurm task per node and one
   `torch.distributed.run` worker group across all nodes.

The multi-node acceptance path is concrete: add a launcher mode that accepts `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `GPUS_PER_NODE` instead of using `--standalone`.
The two-node Cog job uses a 16-rank topology, shared checkpoint storage, and at least one selected
DP or expert-DP group spanning nodes. Wrapping the existing standalone eight-rank script in a
two-node allocation is not accepted because it would create independent worlds.

Local syntax, import, formatter, and `git diff --check` results are lightweight gates. The final
distributed verdict comes from the Cog/Slurm runs.

## Success Criteria

- Internal training code carries one process-group argument, not parallel plain/schedule values.
- No generic path chooses language or dictionary order as a representative module.
- No `assert` protects a runtime topology ambiguity.
- Non-colocated MIMO trains, evaluates, saves, and resumes through the normal training loop.
- Non-colocated, same-topology MIMO has checkpoint parity for the enumerated `torch_dist` modes:
  sync/async, fully-parallel DP/expert-DP save/load, DP-random initialization, and global/local
  non-persistent recovery.
- MIMO FP16 overflow and all step statistics are coherent across grids.
- Ordinary single-grid training behavior and checkpoint APIs remain unchanged.
- Unsupported layouts and checkpoint options fail during startup validation before any collective.
- Generic training-loop code contains no MIMO constants or optimizer-type branches.
