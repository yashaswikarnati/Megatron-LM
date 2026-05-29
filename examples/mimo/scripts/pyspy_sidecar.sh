#!/bin/bash
# py-spy stall-capture sidecar for hetero MIMO training.
#
# Discovers training PIDs on the local node, then writes
# `py-spy dump --native` snapshots at PYSPY_INTERVAL_SEC for PYSPY_DURATION_SEC.
# Designed to be spawned once per node (only on LOCAL_RANK=0) by
# run_hetero_pyspy_wrap.sh and detached so it survives the runner's exec.
#
# py-spy is expected to come from the project venv (${VIRTUAL_ENV}/bin/py-spy,
# version 0.4.2 in m_lm_energon_0506.sqsh + envs/megatron_lm). The sidecar
# fails loudly if py-spy is not on PATH; it will NOT pip-install at runtime.
#
# Env knobs:
#   PYSPY_OUT_DIR        (required, positional $1 or env)
#   PYSPY_INTERVAL_SEC   default 0.5
#   PYSPY_DURATION_SEC   default 3600 (cap; sidecar exits anyway when ranks die)
#   PYSPY_PROC_MATCH     default "train_hetero.py"  (pgrep -f pattern)
#   PYSPY_STARTUP_TIMEOUT default 180s to find training PIDs

set -uo pipefail

OUT="${1:-${PYSPY_OUT_DIR:-}}"
if [[ -z "$OUT" ]]; then
  echo "[pyspy] usage: $0 <out_dir>" >&2
  exit 2
fi

NODE="${SLURMD_NODENAME:-$(hostname -s)}"
INTERVAL="${PYSPY_INTERVAL_SEC:-0.5}"
DURATION="${PYSPY_DURATION_SEC:-3600}"
PROC_MATCH="${PYSPY_PROC_MATCH:-train_hetero.py}"
STARTUP_TIMEOUT="${PYSPY_STARTUP_TIMEOUT:-180}"

NODE_OUT="$OUT/$NODE"
mkdir -p "$NODE_OUT"
LOG="$NODE_OUT/sidecar.log"

log() { echo "[pyspy $(date -u +%H:%M:%S) $NODE] $*" | tee -a "$LOG" >&2 ; }

log "out=$NODE_OUT interval=${INTERVAL}s duration=${DURATION}s match=$PROC_MATCH"

# Resolve py-spy. The venv ships it (py-spy 0.4.2) but the runner uses
# `bash -l` which re-sources /etc/profile and drops ${VIRTUAL_ENV}/bin from
# PATH. Prepend the venv's bin explicitly if VIRTUAL_ENV is set. We never
# pip-install at runtime — that's slow, can hit egress restrictions, and
# obscures provenance. Fail loud if py-spy is still missing.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/py-spy" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:$PATH"
fi
if ! command -v py-spy >/dev/null 2>&1; then
  log "FATAL: py-spy not on PATH"
  log "       PATH=$PATH"
  log "       VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
  log "       expected: \${VIRTUAL_ENV}/bin/py-spy from the project venv"
  exit 1
fi
log "py-spy: $(command -v py-spy) ($(py-spy --version 2>&1))"

# Probe ptrace before busy-looping. If denied, log loudly and exit so the
# training run isn't burdened by repeated failed attaches.
PROBE_OUT=$(py-spy dump --pid $$ --native 2>&1 | head -1 || true)
if echo "$PROBE_OUT" | grep -qiE 'permission denied|operation not permitted|ptrace'; then
  log "FATAL: ptrace denied (yama/scope or missing CAP_SYS_PTRACE): $PROBE_OUT"
  log "       try: sysctl kernel.yama.ptrace_scope=0  OR  --cap-add=SYS_PTRACE in sbatch"
  exit 1
fi
log "ptrace probe OK"

# Wait for training PIDs to appear.
PIDS=""
elapsed=0
while (( elapsed < STARTUP_TIMEOUT )); do
  # -x match against just python procs to avoid catching our own bash. -f to
  # match against the full command line (which includes train_hetero.py).
  PIDS=$(pgrep -u "$USER" -f "$PROC_MATCH" 2>/dev/null \
         | xargs -I{} sh -c 'grep -q python /proc/{}/comm 2>/dev/null && echo {}' \
         | sort -u | tr '\n' ' ')
  [[ -n "${PIDS// /}" ]] && break
  sleep 2
  elapsed=$((elapsed + 2))
done

if [[ -z "${PIDS// /}" ]]; then
  log "FATAL: no PIDs matching '$PROC_MATCH' on $NODE after ${STARTUP_TIMEOUT}s"
  exit 1
fi
log "tracking pids: $PIDS"

# Write a pid -> rank map for manual lookup. RANK/LOCAL_RANK/WORLD_SIZE come
# from /proc/<PID>/environ (torchrun sets them). One file per node so it can
# be grep'd directly: `grep '"rank": 5' ${PYSPY_OUT_DIR}/<node>/pid_rank_map.json`
MAP_FILE="$NODE_OUT/pid_rank_map.json"
{
  echo "{"
  echo "  \"node\": \"$NODE\","
  echo "  \"captured_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"pids\": {"
  first=1
  for pid in $PIDS; do
    env_file="/proc/$pid/environ"
    [[ -r "$env_file" ]] || continue
    rank=$(tr '\0' '\n' < "$env_file" | awk -F= '$1=="RANK"{print $2; exit}')
    local_rank=$(tr '\0' '\n' < "$env_file" | awk -F= '$1=="LOCAL_RANK"{print $2; exit}')
    world_size=$(tr '\0' '\n' < "$env_file" | awk -F= '$1=="WORLD_SIZE"{print $2; exit}')
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 200)
    (( first )) || echo ","
    first=0
    printf '    "%s": {"rank": %s, "local_rank": %s, "world_size": %s, "cmd": "%s"}' \
      "$pid" "${rank:-null}" "${local_rank:-null}" "${world_size:-null}" "${cmd//\"/\\\"}"
  done
  echo ""
  echo "  }"
  echo "}"
} > "$MAP_FILE"
log "wrote pid->rank map: $MAP_FILE"

# Main dump loop. Each iteration fans out py-spy dump in parallel across all
# discovered PIDs, then sleeps INTERVAL. PIDs are re-discovered every 30s in
# case workers restart.
end=$(( $(date +%s) + DURATION ))
last_rediscover=0
while (( $(date +%s) < end )); do
  now=$(date +%s)
  if (( now - last_rediscover > 30 )); then
    NEW_PIDS=$(pgrep -u "$USER" -f "$PROC_MATCH" 2>/dev/null \
               | xargs -I{} sh -c 'grep -q python /proc/{}/comm 2>/dev/null && echo {}' \
               | sort -u | tr '\n' ' ')
    if [[ -n "${NEW_PIDS// /}" && "$NEW_PIDS" != "$PIDS" ]]; then
      log "pid set changed: was='$PIDS' now='$NEW_PIDS'"
      PIDS="$NEW_PIDS"
    fi
    last_rediscover=$now
  fi
  # Bail out if all tracked PIDs are gone.
  alive=0
  for pid in $PIDS; do
    [[ -d "/proc/$pid" ]] && alive=1 && break
  done
  if (( ! alive )); then
    log "all tracked PIDs gone; exiting"
    break
  fi

  ts=$(date +%s.%N)
  for pid in $PIDS; do
    [[ -d "/proc/$pid" ]] || continue
    py-spy dump --pid "$pid" --native \
      > "$NODE_OUT/pid${pid}_${ts}.txt" 2>>"$LOG" &
  done
  wait
  sleep "$INTERVAL"
done
log "loop done; bye"
