"""Cloud-agnostic serverless handler: one mutation/candidate per invocation.

Designed so each evolutionary candidate is an isolated, side-effect-free invocation
that fans out across ephemeral GPU/TPU/CPU workers. Wrap `handle(event)` in whatever
runtime you use (Modal function, RunPod handler, Cloud Run request, AWS Lambda, Azure
Container Apps job).

event = {
  "task_id":   "cholesky_update__gpu__f32",   # OR "target_file" to score a whole file
  "target_file": "jaxlib/gpu/linalg_kernels.cu.cc",   # optional alt to task_id
  "candidate_source": "<full mutated kernel .cc text>",  # optional; staged before build
  "build": true,                               # rebuild+hot-swap, else eval installed .so
  "baseline": { "256": 0.0012, ... }           # optional per-size baseline seconds
}
returns metrics dict (correctness + per-size latency/speedup + build time).
"""
from __future__ import annotations
import os, shutil, tempfile

from jaxbench import registry, runner, build


def _stage_candidate(target_file: str, source: str) -> str | None:
    """Write candidate kernel into the jax checkout; return backup path for restore."""
    dest = os.path.join(build.JAX_REPO, target_file)
    backup = tempfile.mktemp(suffix=".orig")
    shutil.copy2(dest, backup)
    with open(dest, "w") as f:
        f.write(source)
    return backup


def handle(event: dict) -> dict:
    do_build = bool(event.get("build", False))
    baseline = event.get("baseline")

    if event.get("task_id"):
        tasks = [registry.task_by_id(event["task_id"])]
    elif event.get("target_file"):
        tasks = [t for t in registry.tasks() if t.file == event["target_file"]]
    else:
        return {"error": "event needs task_id or target_file", "combined_score": 0.0}
    if not tasks:
        return {"error": "no matching tasks", "combined_score": 0.0}

    target_file = tasks[0].file
    backup = None
    try:
        if event.get("candidate_source"):
            backup = _stage_candidate(target_file, event["candidate_source"])

        built, recs, scores = set(), [], []
        for t in tasks:
            b = do_build and t.build_target not in built
            rec = runner.evaluate_task(t, do_build=b,
                                       baseline=(baseline if event.get("task_id") else None))
            if b and "error" not in rec:
                built.add(t.build_target)
            recs.append(rec); scores.append(runner.score(rec))

        correct = all(r.get("correctness", {}).get("passed", False) for r in recs)
        fitness = (sum(scores) / len(scores)) if (correct and scores) else 0.0
        return {"combined_score": round(fitness, 4), "correct": correct,
                "target_file": target_file,
                "results": {r["task"]: {"score": r.get("score"),
                                        "correct": r.get("correctness", {}).get("passed"),
                                        "perf": r.get("perf")} for r in recs}}
    finally:
        if backup:
            dest = os.path.join(build.JAX_REPO, target_file)
            shutil.copy2(backup, dest); os.remove(backup)
            for so in {t.so_path for t in tasks}:
                try:
                    build.restore(so)
                except Exception:
                    pass


if __name__ == "__main__":  # local smoke: echo a CPU eval
    import json, sys
    ev = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"task_id": "lu__cpu__f64", "build": False}
    print(json.dumps(handle(ev), indent=2))
