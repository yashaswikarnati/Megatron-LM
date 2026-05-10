#!/usr/bin/env bash
# NMFW-464 Phase 1 E2E batch test runner.
#
# Each test runs in its own ``torch.distributed.run`` invocation so global
# singletons (``parallel_state``, NCCL groups, RNG tracker) cannot leak
# between tests. Fails fast on the first non-zero pytest exit so the cog
# batch job returns a meaningful status to the caller.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTEST_TESTS=(
    # Unit tests (mock, fast).
    'tests/unit_tests/test_hyper_comm_grid.py::TestHyperCommGrid'
    'tests/unit_tests/test_hyper_comm_grid.py::TestHyperCommGridAltFactorization'

    # Distributed integration (8 GPUs).
    'tests/unit_tests/test_hyper_comm_grid.py::TestHyperCommGridIntegration'
    'tests/unit_tests/test_process_groups_config.py::TestPGConfigFromHyperCommGrid'

    # MoE-on-schedule.
    'tests/unit_tests/transformer/moe/test_moe_with_hcg_pg.py'

    # MIMO + MoE end-to-end matrix. Each test gets its own torchrun.
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_moe_colocated_8gpu[False]'
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_moe_colocated_8gpu[True]'
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_nemotron_mamba_moe_colocated_8gpu'
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_nemotron_radio_mamba_moe_colocated_8gpu'
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_mamba_moe_non_colocated_8gpu'
    'tests/unit_tests/models/test_mimo_moe_e2e.py::TestMimoMoEColocated::test_mimo_nemotron_radio_mamba_moe_non_colocated_8gpu'
)

results=()
for t in "${PYTEST_TESTS[@]}"; do
    echo
    echo "==================================================================="
    echo "RUN: $t"
    echo "==================================================================="
    # Hash the test id so log directories never collide on long parametrized names.
    log_id="$(printf '%s' "$t" | sha1sum | cut -c1-12)"
    log_dir="${TORCHRUN_LOG_DIR:-/tmp}/${log_id}"
    mkdir -p "$log_dir"

    # Authoritative success/failure comes from pytest's exit code (PIPESTATUS[0]),
    # not from grep matching a "passed" line. The grep is purely for one-line
    # summaries in the batch stdout and is allowed to print nothing.
    set +e
    uv run --no-sync python -m torch.distributed.run \
        --nproc-per-node 8 \
        --redirects=3 --tee=3 \
        --log-dir "$log_dir" \
        -m pytest "$t" -q --tb=line 2>&1 | tee "$log_dir/run.log"
    pytest_rc=${PIPESTATUS[0]}
    set -e

    grep -E '^\[default0\]:.*passed|^\[default0\]:.*failed' "$log_dir/run.log" | tail -3 || true

    if [ "$pytest_rc" -eq 0 ]; then
        results+=("PASS  $t")
    else
        results+=("FAIL  $t (rc=$pytest_rc)")
        echo "==================================================================="
        echo "FAILED: $t (pytest exit code $pytest_rc)"
        echo "Last 80 lines of run.log:"
        tail -80 "$log_dir/run.log"
        echo "==================================================================="
        printf '%s\n' "${results[@]}"
        exit "$pytest_rc"
    fi
done

echo
echo "==================================================================="
echo "ALL TESTS PASSED"
echo "==================================================================="
printf '%s\n' "${results[@]}"
