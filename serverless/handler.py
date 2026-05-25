"""Cloud-agnostic serverless handler: one challenge candidate per invocation.

A candidate evaluation is a pure function, so each evolutionary candidate is an
isolated, side-effect-free invocation that fans out across GPU/TPU/CPU workers. Wrap
`handle(event)` in any runtime (Modal function, RunPod handler, Cloud Run request,
Azure Container Apps job, GCP TPU Cloud Run for the xla_core tier).

event = {
  "challenge_id": "xla_scan_expander",   # which hot-path challenge
  "device": "gpu",                        # gpu | cpu | tpu
  "candidate_source": "<full modified C++ source of challenge.file>",  # optional
  "build": true,                          # rebuild+apply, else score stock on the device
  "baseline": { "8192": 0.0012, ... }     # optional per-size stock latency (seconds)
}
returns {combined_score, correct, tier, perf, ...}.

Tier handling: jaxlib_kernel candidates rebuild one .so and hot-swap it (seconds);
xla_core candidates rebuild the plugin wheel against the editable XLA copy
(--override_repository) — in serverless that means a fresh container/image per
candidate, so the rebuilt wheel is what's installed when this handler runs.
"""
from __future__ import annotations

from jaxbench import challenges, challenge_runner


def handle(event: dict) -> dict:
    cid = event.get("challenge_id")
    if not cid:
        return {"error": "event needs challenge_id", "combined_score": 0.0}
    try:
        c = challenges.by_id(cid)
    except KeyError:
        return {"error": f"unknown challenge {cid}", "combined_score": 0.0}
    return challenge_runner.evaluate(
        c, device=event.get("device", "gpu"),
        do_build=bool(event.get("build", False)),
        candidate_source=event.get("candidate_source"),
        baseline=event.get("baseline"),
    )


if __name__ == "__main__":  # local smoke (CPU, no build)
    import json, sys
    ev = json.loads(sys.argv[1]) if len(sys.argv) > 1 else \
        {"challenge_id": "xla_scan_expander", "device": "cpu", "build": False}
    print(json.dumps(handle(ev), indent=2))
