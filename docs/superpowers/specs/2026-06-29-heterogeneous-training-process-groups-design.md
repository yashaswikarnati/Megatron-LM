# Generic Heterogeneous Training Process-Group Design

**Status:** Final design for the non-colocated DDP implementation on PR #43 at `5408946a31`,
stacked on PR #5516.

## Objective

Refactor the Megatron training loop so ordinary single-grid models and heterogeneous multi-module
models use one process-group carrier throughout setup, model construction, training, evaluation,
optimizer coordination, and checkpointing.

The topology is an ordered mapping from module name to `HyperCommGrid`. Every module owns its grid
instance and PGC. Modules that need identical parallelism on the same ranks simply construct
separate, identically configured grids and PGCs; no extra ownership abstraction is needed.
The implementation target for this effort is non-colocated MCore DDP through the existing MIMO
mock-data path. FSDP, colocated execution, and composite multi-transport checkpoint strategies are
future capabilities that shape the interfaces but are not implemented or tested here.

MIMO must reuse the same model-distribution lifecycle as GPT and Hybrid. Tensor-parallel parameter
attributes, device placement, CPU/meta initialization, mixed precision, FP32 expert-bias
maintenance, FP8 amax correction, DDP wrapping, distributed-optimizer layout, and data-parallel
parameter broadcast must not be reimplemented in an examples-only MIMO path. The shared helper
retains its existing FSDP extension points so adding MIMO FSDP later does not require another
builder path.

Checkpoint parity in this effort is for DDP with the supported `torch_dist` modes. Heterogeneous
topology must not disable async save, persistent workers, fully-parallel load/save,
data-parallel random initialization, or non-persistent recovery when the stock strategy's
single-transport contract is satisfied. FSDP formats, composite multi-transport checkpointing, and
topology resharding remain future work.

## Current Problems

PR #43 carries two related values through the training loop:

- `pg_collection`, a plain `ProcessGroupCollection` used by legacy training helpers; and
- `schedule_pg_collection`, a `MultiModuleProcessGroupCollection` used by the heterogeneous
  schedule.

The language-first/first-entry selection and the later `local_collection` replacement are both
invalid abstractions:

- a multi-module rank may have several encoders even when it does not host the language model;
- selecting the language module or `next(iter(...))` confuses module identity with grid identity;
- dereferencing the union early loses the information needed by the builder, optimizer,
  evaluation, and checkpoint router; and
- an `assert` does not safely guard a runtime topology choice.

The MIMO builder also has a second, incomplete model-distribution path. It unconditionally moves
the outer model to CUDA, constructs DDP configuration from raw arguments, and does not honor the
builder's CPU/meta placement, mixed-precision wrapper, FSDP flags, amax repair, expert-bias
maintenance, or data-parallel random-initialization contract.

Additional correctness gaps are part of this design because the refactor would otherwise preserve
them behind a cleaner API:

1. The builder selects only the first encoder and seeds one encoder role rather than every active
   module.
2. Heterogeneous evaluation falls back to MPU schedule selection.
3. The generic training loop reduces optimizer statistics that `MimoOptimizer` must already make
   coherent across grids.
4. `MimoOptimizer` can update module-local loss scalers before reaching a world overflow decision.
5. The outer `MimoModel` eagerly constructs every child, preventing the builder from independently
   seeding or meta-constructing each child.

## Design Decisions

1. **One orchestration carrier.** Setup, builders, training, evaluation, schedule selection, and
   checkpoint adapters receive:

   ```python
   ProcessGroupCollection | MultiModuleProcessGroupCollection
   ```

2. **The multi-module carrier is rank-local.** It contains every module active on the current
   rank, with one plain collection per module.

3. **No representative module.** Generic code resolves a named module or a named policy role. It
   never chooses the language model, global-last rank, or first dictionary entry as a generic
   representative.

4. **Each module owns its topology.** Even when modules have identical rank placement and parallel
   dimensions, they retain separate grid instances, PGCs, wrappers, optimizers, checkpoint
   namespaces, and initialization seeds.

5. **Plain collections stay below orchestration.** An individual child wrapper, module optimizer,
   low-level collective, and checkpoint transport receives the child module's plain collection or
   an explicit process group.

6. **One shared distribution lifecycle.** GPT, Hybrid, and each active MIMO child call the same
   internal post-build preparation primitive.

7. **The optimizer owns optimizer statistics.** A heterogeneous optimizer returns final coherent
   success, loss-scale, gradient-norm, and zero-count values. The training loop performs no second
   topology-dependent reduction.

8. **FSDP stays an extension point.** This effort does not implement or test MIMO FSDP. The shared
   lifecycle and module-local PGC plumbing must remain wrapper-neutral, and MIMO code must not add
   new DDP-only assumptions that make later FSDP support require another architecture.

9. **Ordinary behavior stays unchanged.** A normal run still uses a plain collection materialized
   from MPU and follows existing schedule, reduction, wrapper, and checkpoint behavior.

10. **Ambiguity fails before collectives.** Topology and feature validation uses `ValueError` or
    `NotImplementedError`, never `assert`, for user-visible runtime restrictions.

## Process-Group Carrier

`MultiModuleProcessGroupCollection` remains a mapping from module name to the plain collection for
that module on the current rank. Its policy field is generalized from
`language_model_module_name` to `loss_module_name`. It also carries the immutable global
`module_order` needed for deterministic routing. MIMO sets the loss module to its language module;
generic training code only understands that this module owns the scalar training loss.

For two encoders with identical parallelism on the same ranks, the rank-local carrier still contains
two distinct PGCs:

```python
# Encoder-grid rank.
MultiModuleProcessGroupCollection(
    module_pgs={
        "vision": vision_pgc,
        "audio": audio_pgc,
    },
    loss_module_name="language",
    module_order=("vision", "audio", "language"),
)

# Language-grid rank.
MultiModuleProcessGroupCollection(
    module_pgs={"language": language_pgc},
    loss_module_name="language",
    module_order=("vision", "audio", "language"),
)
```

`loss_module_name` is global policy metadata and therefore may name a module absent from the
current rank's mapping. `MultiModuleProcessGroupCollection.__post_init__` validates local shape and
the global module order. `HeteroTopology` performs global grid and loss-module validation; a
rank-local object cannot prove global topology by itself.

The collection exposes explicit operations:

- `get_module_collection(module_name)` for module-specific work;
- `get_loss_module_collection()` returning the loss module's plain collection or `None` on ranks
  outside that module.

`local_collection` is removed. A rank with several modules has several module-owned collections,
so generic code must name the module whose groups it needs.

The rank-local mapping is a lookup structure, not an execution-order contract. All construction,
wrapper, optimizer, grad-finalization, and checkpoint operations filter the explicit serialized
`module_order`. Ranks participating in the same module therefore issue collectives in the same
order; dictionary insertion order is not NCCL ordering policy.

## Topology Ownership

Topology ownership stays module-keyed:

```python
module_to_grid_map: dict[str, HyperCommGrid]
module_pgs: dict[str, ProcessGroupCollection]
module_order: tuple[str, ...]
```

`HeteroTopology` creates and destroys one grid and one PGC for every module in `module_order`.
Identically configured modules still get separate instances. `MimoModelConfig`, `RankRole`, and the
dependency DAG consume the existing module-keyed map; scalar `_encoder_module_name()` selection is
removed from topology construction. The existing mock provider remains one iterator and may be
adjusted only for the fields required by the supported E2E fixture. Checkpoint metadata stores each
module's grid dimensions and rank span directly.

## Training-Loop Plumbing

The sole argument is named `pg_collection` and accepts the union.
`schedule_pg_collection` is deleted from `pretrain()`, `train()`, `train_step()`, evaluation, and
all callers; no compatibility alias or normalizer is retained. Internal calls, local variables,
and helper signatures carry only `pg_collection`.

When no carrier is supplied, ordinary initialization runs and constructs a plain collection from
MPU. When a multi-module carrier is supplied:

- the union remains intact through model setup, training, evaluation, and checkpoint routing;
- process-global initialization does not select a model module for RNG policy;
- the MIMO builder seeds each active child immediately before constructing that child using its
  explicit module group and stable module-specific seed offset; and
- initialization work that is actually module-specific is moved behind the builder or invoked per
  module rather than supplied one arbitrary group set.

The non-colocated validator reasons over declared grids rather than assuming one grid per module.
For the current DDP schedule it requires:

- no encoder grid overlaps the language grid, and the union of participating grid spans
  covers the intended coordination ranks;
- every active module key maps to the exact plain collection created for its declared grid;
- one loss module is named consistently;
- the current reporting-rank compatibility requirements are satisfied; and
- the carrier is paired with a `MultiModulePipelineCommunicator`.

No carrier or builder API requires one module or one grid per rank. The current E2E acceptance
topology remains the existing non-colocated encoder/language layout supported by the bridge
schedule and one mock iterator per rank. Broader module/grid mappings remain representable, but a
layout outside the current schedule contract fails before schedule selection. This effort does not
add module-specific iterators, microbatch state, or dataloader checkpoint state.

`validate_args()` gains an explicit `data_parallel_size_override`. It validates the override
against the language-grid specification and applies it instead of recomputing DP size from the
process-wide TP/PP/CP arguments. Global/evaluation batch fields and the microbatch calculator are
then initialized from that value. The post-validation reassignment currently in the MIMO entry
point is removed.

## Shared Model Distribution Lifecycle

`megatron.training.models.dist_utils` gains one private primitive for already-built homogeneous
chunks:

```python
def _prepare_model_chunks_for_distributed(
    model_chunks: list[MegatronModule],
    transformer_config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    ddp_config: DistributedDataParallelConfig | None = None,
    *,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    mixed_precision_wrapper: Callable | None = Float16Module,
) -> list[MegatronModule]: ...
```

It performs, in order:

1. default tensor-parallel parameter attributes;
2. module-local parameter reporting;
3. CUDA, CPU, meta-device materialization, or FSDP2-deferred placement;
4. mixed-precision wrapping;
5. FP32 expert-bias maintenance;
6. meta-device materialization where required;
7. FP8 amax-history correction;
8. DDP, Megatron FSDP, or Torch FSDP2 wrapping;
9. distributed-optimizer full-parameter layout and bucket setup; and
10. data-parallel random-initialization broadcast through the created child wrapper.

`wrap_with_ddp=False` skips only step 8 and the wrapper-dependent broadcast. Placement,
mixed-precision handling, expert-bias maintenance, and amax repair still run.

`unimodal_build_distributed_models()` remains signature-compatible. It owns VP-stage construction
and enters the meta-device context before construction when configured. After construction and its
existing hook boundary, it calls the shared primitive. GPT and Hybrid remain thin delegates and
retain their existing defaults. The post-build primitive can materialize an already-meta child;
it cannot retroactively make an eagerly built child a meta-device construction.

The helper accepts only a plain module-local collection. It does not understand MIMO,
`MultiModuleProcessGroupCollection`, topology objects, bridges, or checkpoint namespaces, and it is
not exported as a new public training API.

### MIMO Builder

`ModelBuilder.build_distributed_models()` is an orchestration boundary and accepts the union.
Ordinary builders require a plain collection. `MimoModelBuilder` requires a multi-module
collection and uses it as the authoritative source for active child groups.

`MimoModel` gains a deferred-child construction path. Its existing eager construction remains the
default for direct/legacy callers, while `MimoModelBuilder.build_distributed_models()` uses a
topology-only outer shell. The MIMO sequence is:

1. Validate carrier keys and plain-collection identities against the active subset of
   `HeteroTopology`.
2. Build one empty outer `MimoModel` shell without moving it to CUDA or constructing children.
3. Iterate every locally active child in explicit `module_order`. This is a module iteration, not
   a unique-PGC iteration.
4. Seed with the child's stable explicit seed offset, enter that child's CPU/meta construction
   context, construct the raw child, and attach it to the shell. Python hashing is not used.
5. Run the outer pre-wrap hook once after all local raw children are attached.
6. Apply module freezing before distributed-optimizer layout is computed.
7. Select the child's explicit transformer config and mixed-precision wrapper policy.
8. Clone the caller's DDP/FSDP config for the child, then apply only documented module-specific
   overrides. Cloning is required because bucket-size calculation mutates the config.
9. Call `_prepare_model_chunks_for_distributed([child], ...)` with the child's exact plain
   collection and reattach the returned wrapper to the outer model.
10. Run the outer post-wrap hook once, configure heterogeneous runtime callbacks, and store the
    complete multi-module carrier on `MimoModel`.

Every active child executes steps 4-9 independently and retains its own PGC, wrapper, layout,
optimizer entry, and checkpoint prefix.

Language modules use the normal mixed-precision wrapper. Encoders may supply the MIMO encoder
wrapper needed to preserve bridge-output precision. That is module policy passed into the shared
lifecycle, not a duplicate lifecycle. `None` and caller-supplied wrappers follow the same builder
contract and receive focused tests.

Generic pre/post builder hooks execute once around the outer `[mimo_model]` boundary as described
above. Child-specific precision and freezing policy does not masquerade as repeated generic hooks.

Because the outer model is intentionally unwrapped, the builder installs composite
`no_sync_func`, `grad_sync_func`, and `param_sync_func` callbacks on the outer training config.
Those callbacks traverse child wrappers in explicit module order. Heterogeneous gradient
finalization receives the carrier, coordination group, and module order directly; it does not
capture `HeteroTopology` or rely on `ModuleDict` insertion order.

`MimoBuildConfig` contains typed build values rather than a full `argparse.Namespace`. A temporary
non-serialized topology handle may remain until a general builder runtime-context API exists, but
it cannot duplicate the authoritative rank-local process-group mapping.

The initial MIMO builder continues to reject virtual-pipeline parallelism because its outer return
value cannot represent independently chunked child lists. Ordinary pipeline parallelism remains
module-local and uses each child's topology.

## DDP and FSDP Contract

The outer `MimoModel` is not itself data-parallel wrapped. Every active child is independently
prepared and wrapped.

`DistributedDataParallel` receives the child's exact plain collection. The shared lifecycle owns
wrapper construction, bucket/layout calculation, overlap options, and DP-random parameter
broadcast. MIMO's composite outer callbacks expose the child DDP operations to the ordinary
training loop.

This effort does not implement or test Megatron FSDP or Torch FSDP2 for MIMO. The design preserves
the following extension constraints:

- the preparation helper keeps its existing wrapper-selection flags and accepts only the child's
  plain PGC;
- MIMO orchestration and checkpoint namespaces do not depend on `isinstance(..., DDP)` where a
  common wrapper operation is intended;
- Torch FSDP2 must eventually receive the child's explicit DP-CP group instead of MPU fallback;
  and
- wrapper-specific gradient scaling, parameter initialization/broadcast, optimizer state, and
  checkpoint targets must be implemented and tested before MIMO FSDP is claimed.

Until that follow-up lands, FSDP is outside this effort's acceptance matrix. It is not a reason to
create a second MIMO distribution path or a new MIMO-specific startup rejection. The builder keeps
the wrapper flags and module-local PGC plumbing aligned with the shared helper, but this effort does
not claim FSDP correctness beyond existing stock validation.

## Training, Evaluation, and Logging

`train()` and `train_step()` pass the same union object to schedule selection, the selected
forward/backward schedule, metric helpers, and checkpoint adapters. The separate
`model_pg_collection` and `schedule_pg_collection` internal variables are removed.

`evaluate()` and `evaluate_and_print_results()` receive the same carrier and communicator as
training. A heterogeneous run with `eval_iters > 0` therefore uses the bridge schedule rather than
MPU globals. `collect_non_loss_data` still requires all heterogeneous ranks to participate; only
the explicit reporting owner processes the result.

The non-colocated communicator does not choose `current_stage` from the first mapping entry. For
the supported E2E layout it derives the stage from explicit topology metadata and validates the
local module role before schedule collectives. This effort keeps the existing bridge schedule and
does not add a new multi-stage scheduler.

Valid schedule pairs are checked before collectives:

- a plain collection with an ordinary communicator or the ordinary MPU fallback; and
- a multi-module collection with a `MultiModulePipelineCommunicator`.

Loss reduction and terminal-stage detection use `get_loss_module_collection()`. Encoder-only ranks
receive `None` and do not fabricate a primary collection. Full-validation and time-limit control
use the explicit world coordination group.

Ordinary optimizer statistics retain their current reduction when the carrier is plain. The
multi-module path accepts already-coherent values from its optimizer and performs no second
reduction. Generic training code does not branch on `MimoOptimizer`; this is part of the
multi-module optimizer contract.

Metrics resolve only the group required by that metric. A metric without defined heterogeneous
semantics fails clearly instead of reducing on the language or first encoder grid.

### Mock Data Contract

The E2E path continues using the existing mock dataloader and one iterator per rank. Its batch
already carries language fields and a `modality_inputs` mapping. The mock provider may be modified
to populate the fields required by the supported MIMO topology, but this effort does not introduce
per-encoder iterators, independent microbatch schedulers, or a new production dataloader API.

The mock iterator remains intentionally stateless for checkpointing. Resume correctness is proven
from model, optimizer, scheduler/scaler, RNG, iteration, and consumed-sample state; no mock-loader
state is saved or restored.

## MIMO Optimizer Coordination

`MimoOptimizer` owns one inner optimizer for every active child module, including several encoders
on the same rank. Each inner optimizer is built from the child wrapper and that child's exact plain
collection. Identical grid layouts do not combine optimizers. The topology-created PGC includes
every optimizer-required group, including `intra_dist_opt`; `get_mimo_optimizer()` no longer
reconstructs a second collection from `HyperCommGrid`. Caller options such as `config_overrides`
and `use_gloo_process_groups` are forwarded to every child optimizer.

The language span is disjoint from every supported encoder span. Their participating-rank union
covers world, so `torch.distributed.group.WORLD` is the explicit optimizer coordination group.
Supporting a subset of world requires an explicit coordination group but no training-loop change.

The step sequence is:

1. Every active inner optimizer copies/unscales gradients and detects local overflow through
   `prepare_grads(update_grad_scaler=False)`.
2. MIMO reduces overflow across the coordination group.
3. Every active mixed-precision optimizer updates its full scaler state exactly once from the same
   global decision.
4. On overflow, every rank returns the same unsuccessful result before norm collectives.
5. Per-module norm contributions occupy a canonical module-keyed vector, reduce across the
   coordination group, and combine into one global norm.
6. Each inner optimizer clips with that global norm while preserving its precision-aware gradient
   handling.
7. Inner optimizers step. A false result after the global overflow decision is an invariant
   violation.
8. Requested zero counts use the same canonical vector/reduction pattern.

The optimizer API change is backward-compatible:

```python
def prepare_grads(self, *, update_grad_scaler: bool = True) -> bool: ...
def update_grad_scaler(self, found_inf: bool) -> None: ...
```

`MixedPrecisionOptimizer` updates scale, growth tracker, and hysteresis tracker.
`FP32Optimizer` implements a no-op. `ChainedOptimizer` forwards both operations. Existing callers
retain current behavior by omitting the keyword.

Every module optimizer checkpoints its own scaler state. Resume verifies complete scaler-state
equality across the coordination group before the first step. Divergent legacy state fails
explicitly rather than selecting one module's scaler.

The initial MIMO optimizer preserves one global norm. `grad_norm_skip_threshold`, registered
separate grad-norm groups, and multiple distributed-optimizer instances remain explicitly outside
this optimizer change because they require their own cross-grid decision semantics.

## Checkpointing

The training-layer checkpoint adapter receives the union and produces one composite state dict and
one checkpoint request. Low-level save/load functions remain plain-group APIs; they do not learn
MIMO module names.

Checkpoint state separates three concerns:

1. **Module state.** Model and optimizer entries are keyed by stable module name. Every module
   remains an independent entry.
2. **Grid transport.** Modes whose stock strategy accepts one parallelization group require one
   local module/transport PGC per rank in the current supported topology.
3. **Process RNG state.** Python, NumPy, Torch, CUDA, and CUDA-tracker state is process-owned unless
   a module explicitly creates a named tracker. It is saved exactly once per global rank under a
   topology-independent namespace and restored once. No child grid is selected as an RNG
   representative. Per-module seed offsets guarantee independent parameter initialization; they
   do not create persistent per-module runtime RNG streams.

A single-group stock strategy validates the supported rank role and resolves its PGC with
`get_module_collection(active_module_name)`, not dictionary iteration. A future multi-transport
strategy receives explicit per-module routes. No generic representative-module accessor is
introduced for this purpose.

The adapter supplies explicit TP/PP/DP/DP-CP/expert-DP groups to save and load. Synchronous
finalization derives ranks and sizes from those groups. Async requests capture the resolved
integer ranks/sizes and immutable module-routing metadata before enqueueing; callbacks do not
rediscover topology from MPU or a retained arbitrary PGC.

Fully-parallel load uses the validated local transport group consistently for group size and object
exchange; it never expands to world implicitly. Local non-persistent save/restore receives that
module's DP-CP group. Global non-persistent finalization uses its selected save directory for tracker
retention and cleanup.

Plain persistent sync/async and global non-persistent `torch_dist` use one world-coordinated
request and are supported for the current non-colocated mock topology. Stock fully-parallel and
tensor-aware local-recovery strategies, however, accept one parallelization group
for the entire state/request. Any rank with distinct local PGCs—whether several encoders share rank
placement or modules are fully colocated—cannot be routed by making multiple low-level calls: that
would duplicate common state, async requests, tracker transitions, and finalizers. Until the
general multi-transport phase lands, those specific modes reject that topology before collectives.
General support requires one world-coordinated composite plan with per-module routes but exactly one
common-state write, tracker transition, async lifecycle, and finalizer.

Common checkpoint metadata stores the complete module topology: module name, rank span, TP, PP,
CP, DP, EP, and expert-TP dimensions, plus stable canonical module order. Resume
validates runtime metadata before producing model, optimizer, or RNG sharded state. Same-topology
resume is required; heterogeneous topology resharding fails before checkpoint collectives.

The adapter is used by initial load, interval save, signal save, duration/iteration exit save, and
final save. RL reference reload, MoE upcycling, and format conversion must either use the same
adapter or reject the multi-module carrier explicitly before collectives.

The mock external dataloader is stateless. Saving or restoring its private iterator state is not
part of this PR; consumed-sample accounting and ordinary training state are restored.

### Data-Parallel Random Initialization

Each active module is seeded independently using its explicit group and stable module seed offset.
After DDP wrapping, the shared distribution primitive calls `broadcast_params()` when
`data_parallel_random_init=True`; each child wrapper owns the correct dense and expert groups. RNG
save/load remains process-owned as described above. Future FSDP support must choose a
wrapper-correct synchronization point without changing MIMO orchestration.

### Checkpoint Parity Matrix

| Mode | Heterogeneous requirement |
| --- | --- |
| Persistent synchronous `torch_dist` | Supported for wrappers supported by the stock format |
| Persistent async save and persistent worker | Supported when the selected stock format supports it |
| Fully-parallel DP and expert-DP save/load | Supported with the rank's validated single transport PGC |
| Stock fully-parallel exchange algorithms | Preserve stock validation and group boundaries |
| `data_parallel_random_init` and RNG restore | Supported per module/grid |
| Global non-persistent `torch_dist` | Supported for the current non-colocated mock topology |
| Local non-persistent sync/async and replication | Supported for one local transport PGC when stock NVRx is available |
| Cached checkpoint structure/strategy reuse | Supported |
| Model, optimizer, scheduler, scaler, RNG, iteration, consumed samples | Restored |
| Megatron FSDP or Torch FSDP2 format | Outside this effort; architecture only |
| Legacy checkpoint format | Rejected before model/checkpoint collectives |
| Topology-changing resume | Rejected before checkpoint collectives |
| Colocated distinct-PGC fully-parallel/local recovery | Requires a future composite strategy |

## General Topology Boundaries

| Layout | Carrier | Design status |
| --- | --- | --- |
| Ordinary single-grid model | Plain collection | Unchanged |
| Legacy colocated MIMO using ordinary schedule | Existing plain path | Unchanged |
| Existing non-colocated mock MIMO layout | Rank-local multi-module collection | Required E2E target |
| Other module-specific encoder-grid layouts | Rank-local multi-module collection | Representable; focused carrier/builder coverage only |
| Colocated heterogeneous bridge execution | Rank-local multi-module collection | Outside this effort |
| Several local checkpoint transport PGCs | Rank-local multi-module collection | Requires a future composite strategy |

Carrier and builder code do not collapse a colocated rank to a representative module. End-to-end
colocated bridge execution is nevertheless not implied by representation. The current
`MultiModulePipelineCommunicator` has one scalar `current_stage` and chooses it from a mapping
entry, while the outer colocated forward executes multiple DAG stages on one rank. Supporting this
cleanly requires either an ordinary/no-pipeline schedule that understands the union or a dedicated
colocated multi-stage schedule. It also requires the composite checkpoint plan described above for
multi-transport fully-parallel/local modes.

The current implementation rejects a carrier containing both language and encoder stages on one
rank before bridge schedule selection rather than using dictionary order. The legacy colocated
path remains unchanged. Future schedule/checkpoint extensions use the same carrier and builder
instead of adding another process-group API.

Generic training code imports only generic collection and communicator interfaces. MIMO module
names, topology policy, wrapper selection, optimizer coordination, and bridge semantics remain in
MIMO-specific code.

## Review-Comment Scope

The implementation addresses the PR feedback that intersects this design:

- replace language/first-module and `local_collection` selection;
- carry one union argument internally instead of plain and schedule collections;
- make the MIMO builder consume the passed carrier;
- preserve and focused-test the module-to-grid mapping with one grid/PGC per module;
- defer child construction for per-module seed and meta-device semantics;
- remove the duplicated MIMO distribution lifecycle;
- store the complete rank-local carrier on `MimoModel`;
- remove raw parsed arguments from `MimoBuildConfig`;
- make checkpoint and RNG namespaces explicit and wrapper-aware;
- honor or explicitly validate every builder option; and
- preserve mock iterator scope and CLI-controlled evaluation/rerun behavior.

No GitHub review replies or thread resolutions are part of implementation unless explicitly
requested.

## Implementation Scope

1. Make module order explicit while retaining one grid and PGC per module.
2. Carry one union through pretrain, setup, train, train-step, evaluation, schedules, logging, and
   checkpoint adapters.
3. Extract the post-build distribution primitive and prove GPT/Hybrid behavior is unchanged.
4. Add deferred MIMO child construction and prepare every active child with its own PGC.
5. Install composite outer runtime callbacks and coordinate optimizer overflow, scaler, norm, and
   zero counts.
6. Deliver DDP persistent sync/async, fully-parallel DP/expert-DP, global/local non-persistent
   recovery, and DP-random initialization for the supported non-colocated mock topology.
7. Make the existing mock dataloader drive train, evaluation, save-at-iteration-2, and
   resume-to-iteration-3 without adding iterator checkpoint state.

FSDP, a new colocated schedule, composite multi-transport checkpointing, and new data-iterator
architectures are not implementation milestones in this effort. Their extension constraints stay
documented above so current code does not close those paths accidentally.

Each increment uses TDD and receives correctness and adversarial review before the next one.

## Verification Strategy

Focused unit and distributed tests prove:

- named/loss-module resolution never depends on insertion order;
- one union object reaches setup, train, train-step, evaluation, schedule, and checkpoint routes;
- ordinary plain-collection behavior is unchanged;
- carrier/builder unit cases cover multiple module-owned grids without selecting a representative
  module;
- the communicator derives the supported layout's stage from topology metadata rather than the
  first mapping entry;
- every active child is prepared in canonical order and receives its own wrapper, config,
  optimizer, and checkpoint namespace;
- cloned per-module DDP configs prevent bucket-size or overlap-policy mutation from leaking;
- TP attributes, placement, mixed precision, expert-bias maintenance, amax correction, wrapper
  creation, and DP broadcast occur in the shared lifecycle and in the correct order;
- `wrap_with_ddp=False` skips wrapping without skipping the other preparation phases;
- DDP receives each child's exact groups with no MPU fallback;
- data-parallel random initialization broadcasts within each declared module grid;
- heterogeneous evaluation with `eval_iters > 0` uses the same carrier and communicator as
  training;
- overflow injected on only the encoder grid produces identical scaler state and skipped-step
  results on all ranks;
- success, global norm, and zero count are already coherent when returned to the training loop;
- single-group checkpoint strategies reject a rank with distinct transport PGCs before
  collectives, while carrier/build tests still prove no primary module is selected.

Checkpoint acceptance saves at iteration 2, loads every model and optimizer entry, scheduler,
scaler, RNG, iteration, and consumed-sample count, runs iteration 3, and advances the tracker to 3.
Floating tensors use existing dtype-aware distributed-checkpoint tolerances; integer and serialized
state use exact equality. Loss is not compared against an uninterrupted mock-dataloader run because
the intentionally stateless mock iterator restarts its private generator.

The MIMO persistent matrix covers synchronous and repeated async saves, persistent workers, one
fully-parallel broadcast exchange over DP, a separate nontrivial expert-DP case, and cached
structure reuse. Core distributed-checkpoint tests retain the full exchange-algorithm matrix.
Async success advances the tracker only after finalization; failure or abort does not advance it.

The generic global non-persistent round-trip baseline must be enabled and pass before MIMO global
recovery is claimed. MIMO adds one representative local NVRx recovery case after core tests cover
the algorithm matrix. Replication coverage removes a primary shard and proves recovery from a
peer; merely enabling replication is insufficient.

Wrapper acceptance in this effort is a DDP `torch_dist` round trip. No FSDP job is part of the
matrix.

Topology metadata tests independently change module set, order, rank span, TP, PP, CP, DP, EP, and
expert-TP and require deterministic rejection before checkpoint collectives.

Cluster validation uses CW DFW through `cog`:

1. run `cog doctor --cluster-name cw-dfw-dev`;
2. dry-run every submission and inspect workspace sync, image, environment, task layout, and Slurm
   command;
3. run focused 8-GPU DDP topology, builder, schedule, optimizer, and checkpoint tests with
   `NCCL_MAX_NCHANNELS=1`, `NCCL_NVLS_ENABLE=0`, and
   `CUDA_DEVICE_MAX_CONNECTIONS=1`;
4. run the existing heterogeneous mock training topology with evaluation enabled, modifying only
   the mock provider/batch fields needed by the normal training loop;
5. run DDP save-at-2/resume-to-3 jobs for the claimed checkpoint modes; and
6. run one multi-node DDP async/fully-parallel resume job using one Slurm task per node and one
   `torch.distributed.run` worker group across all nodes, a shared checkpoint path, and a declared
   DP or expert-DP transport group whose asserted membership spans both nodes.

The multi-node launcher accepts `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and
`GPUS_PER_NODE`. Independent per-node `--standalone` worlds are not accepted.

Local syntax, import, formatter, and `git diff --check` checks are lightweight gates. Distributed
correctness is decided by Cog/Slurm runs.

## Success Criteria

- Internal training code carries one process-group carrier, never parallel plain/schedule values.
- No generic path chooses language, global-last, or dictionary order as a representative module.
- Each module owns its explicit grid and PGC, even when another module uses identical topology.
- GPT, Hybrid, and MIMO children share one distribution lifecycle.
- CPU/meta placement, mixed precision, expert bias, FP8 amax, DDP, distributed-optimizer
  layout, and DP-random broadcast have unimodal parity.
- The existing non-colocated mock MIMO topology trains and evaluates through the normal loop.
- MIMO returns coherent overflow, scaler, success, norm, and zero-count values across grids.
- DDP uses exact module groups and module-keyed checkpoint state.
- Persistent sync/async, fully-parallel DP/expert-DP, and global/local non-persistent checkpointing
  have `torch_dist` parity for the supported non-colocated mock topology.
- Ordinary single-grid training behavior remains unchanged.
- Unsupported combinations within the claimed DDP/checkpoint matrix fail before collectives and
  name the unsupported operation.
- Generic training-loop code contains no MIMO constants or optimizer-type branches.
