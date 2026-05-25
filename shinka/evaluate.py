#!/usr/bin/env python3
"""ShinkaEvolve evaluation entrypoint for JaxBench.

ShinkaEvolve mutates the C++/CUDA inside a file's `// EVOLVE-BLOCK-START/END` region,
then calls this script to score the candidate. We run the validated inner loop:

    build the one affected extension .so (clang device compiler)
      -> hot-swap it into the venv
      -> correctness gate (residuals over the N-sweep; wrong => fitness 0)
      -> latency sweep -> speedup vs the recorded baseline
      -> emit fitness JSON on stdout

Fitness = 0.0 if any task for this file fails correctness, else the mean speedup over
all of that file's tasks. Shinka maximises fitness.

Usage:
    python shinka/evaluate.py --file jaxlib/gpu/linalg_kernels.cu.cc \
                              --baseline shinka/baseline.json [--no-build]
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaxbench import registry, runner  # noqa: E402


def tasks_for_file(path: str):
    return [t for t in registry.tasks() if t.file == path]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="mutated jaxlib source file (registry key)")
    ap.add_argument("--baseline", default="shinka/baseline.json",
                    help="JSON of baseline latencies {task_id: {N: seconds}}")
    ap.add_argument("--no-build", action="store_true",
                    help="skip rebuild/hot-swap (e.g. baseline run on unmutated tree)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    tlist = tasks_for_file(a.file)
    if not tlist:
        print(json.dumps({"error": f"no tasks for file {a.file}", "fitness": 0.0}))
        return 1

    baseline = {}
    if os.path.exists(a.baseline):
        baseline = json.load(open(a.baseline))

    built = set()
    records, scores = [], []
    for t in tlist:
        # build+swap once per build target (shared by several tasks of the file)
        do_build = (not a.no_build) and (t.build_target not in built)
        rec = runner.evaluate_task(t, do_build=do_build, baseline=baseline.get(t.id))
        if do_build and "error" not in rec:
            built.add(t.build_target)
        records.append(rec)
        scores.append(runner.score(rec))

    correct = all(r.get("correctness", {}).get("passed", False) for r in records)
    fitness = (sum(scores) / len(scores)) if (correct and scores) else 0.0
    result = {"file": a.file, "n_tasks": len(tlist), "correct": correct,
              "fitness": round(fitness, 4),
              "per_task": {r["task"]: r.get("score", 0.0) for r in records}}
    out = json.dumps(result, indent=2)
    print(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
    # Shinka reads fitness from stdout; non-zero exit signals a hard failure
    return 0 if correct else 2


if __name__ == "__main__":
    raise SystemExit(main())
