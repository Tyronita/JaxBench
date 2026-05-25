"""End-to-end JaxBench evaluation: the loop ShinkaEvolve drives per mutation.

    mutate kernel file  (done by ShinkaEvolve, outside this module)
      -> build_target   (targeted .so, clang device compiler)
      -> hot_swap       (replace installed extension .so)
      -> correctness    (gate; residuals over the N-sweep)
      -> bench          (latency/GFLOP/s sweep)
      -> speedup vs the recorded baseline
      -> JSON record

`evaluate_task` returns a dict; `score` collapses it to a single fitness number for
the evolutionary loop (0 if incorrect, else mean speedup).
"""
from __future__ import annotations
import json, os, time
from dataclasses import asdict

from . import registry, correctness, bench, build, metrics

RESULTS_DIR = os.environ.get("JAXBENCH_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))


def evaluate_task(task, *, do_build: bool = False, baseline: dict | None = None,
                  seed: int = 0) -> dict:
    rec = {"task": task.id, "op": task.op, "file": task.file, "device": task.device,
           "dtype": task.dtype, "build_target": task.build_target, "ts": time.time()}

    if do_build:
        log = os.path.join(RESULTS_DIR, f"build_{task.id}.log")
        br = build.build_target(task.build_target, log_path=log)
        rec["build"] = {"rc": br.rc, "wall_s": round(br.wall_s, 2),
                        "elapsed_s": br.elapsed_s}
        if br.rc != 0:
            rec["error"] = "build failed"; return rec
        dest = build.hot_swap(br.so_src, task.so_path)
        rec["hot_swapped"] = dest

    cor = correctness.check_task(task, seed)
    rec["correctness"] = {"passed": cor.passed, "worst_residual": cor.worst_residual,
                          "tol": cor.tol, "per_size": cor.per_size}
    if not cor.passed:
        rec["score"] = 0.0; return rec

    ms = bench.sweep(task, seed)
    rec["perf"] = [asdict(m) for m in ms]
    if baseline:
        sp = []
        for m in ms:
            b = baseline.get(str(m.n))
            if b:
                sp.append(metrics.speedup(b, m.latency_ms / 1e3))
        rec["speedup_per_size"] = {str(m.n): s for m, s in zip(ms, sp)}
        rec["score"] = float(sum(sp) / len(sp)) if sp else 1.0
    else:
        rec["score"] = 1.0  # baseline run: speedup defined as 1.0
    return rec


def score(rec: dict) -> float:
    return float(rec.get("score", 0.0))


def save(rec: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"result_{rec['task']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run a JaxBench task end-to-end.")
    ap.add_argument("task_id")
    ap.add_argument("--build", action="store_true", help="rebuild + hot-swap the kernel .so first")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t = registry.task_by_id(a.task_id)
    rec = evaluate_task(t, do_build=a.build, seed=a.seed)
    print(json.dumps(rec, indent=2))
    print("saved:", save(rec))


if __name__ == "__main__":
    main()
