#!/usr/bin/env python3
"""ShinkaEvolve evaluation program for JaxBench (matches the Shinka eval contract).

ShinkaEvolve calls this as:

    evaluate.py --program_path <candidate kernel file> --results_dir <dir>

and reads back, from `results_dir`:

    metrics.json  -> {"combined_score": <fitness>, ... }   (Shinka maximises combined_score)
    correct.json  -> {"correct": <bool>, "error": <str>}

The "program" Shinka evolves is one jaxlib kernel file (its EVOLVE-BLOCK region). We:
  1. copy the candidate into the jax checkout (overwriting that kernel),
  2. rebuild only the affected extension .so (clang device compiler),
  3. hot-swap the .so into the venv,
  4. run the correctness gate over that file's tasks (fail => combined_score 0),
  5. measure latency and compute mean speedup vs the recorded baseline,
  6. write metrics.json / correct.json, and restore the original kernel.

The target file is taken from $SHINKA_TARGET_FILE, else inferred from the candidate's
basename against the registry.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaxbench import registry, runner, build  # noqa: E402


def _resolve_target_file(program_path: str) -> str | None:
    env = os.environ.get("SHINKA_TARGET_FILE")
    if env:
        return env
    base = os.path.basename(program_path)
    for t in registry.tasks():
        if os.path.basename(t.file) == base:
            return t.file
    return None


def _write(results_dir: str, metrics: dict, correct: bool, error: str):
    rp = Path(results_dir); rp.mkdir(parents=True, exist_ok=True)
    (rp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (rp / "correct.json").write_text(json.dumps({"correct": correct, "error": error}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program_path", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()

    target_file = _resolve_target_file(a.program_path)
    if not target_file:
        _write(a.results_dir, {"combined_score": 0.0}, False,
               f"could not map {a.program_path} to a registry file (set SHINKA_TARGET_FILE)")
        return 2

    tlist = [t for t in registry.tasks() if t.file == target_file]
    dest = os.path.join(build.JAX_REPO, target_file)
    backup = dest + ".shinka.orig"
    baseline = {}
    bpath = os.path.join(os.path.dirname(__file__), "baseline.json")
    if os.path.exists(bpath):
        baseline = json.load(open(bpath))

    try:
        # stage the candidate kernel into the jax checkout
        if not a.no_build:
            shutil.copy2(dest, backup)
            shutil.copy2(a.program_path, dest)

        built, records, scores = set(), [], []
        for t in tlist:
            do_build = (not a.no_build) and (t.build_target not in built)
            rec = runner.evaluate_task(t, do_build=do_build, baseline=baseline.get(t.id))
            if do_build and "error" not in rec:
                built.add(t.build_target)
            records.append(rec); scores.append(runner.score(rec))

        build_failed = any(r.get("error") == "build failed" for r in records)
        correct = (not build_failed) and all(
            r.get("correctness", {}).get("passed", False) for r in records)
        fitness = (sum(scores) / len(scores)) if (correct and scores) else 0.0

        metrics = {
            "combined_score": round(fitness, 4),       # Shinka maximises this
            "mean_speedup": round(fitness, 4),
            "n_tasks": len(tlist),
            "per_task_score": {r["task"]: r.get("score", 0.0) for r in records},
            "build_seconds": {r["task"]: r.get("build", {}).get("wall_s")
                              for r in records if r.get("build")},
            "target_file": target_file,
        }
        err = "build failed" if build_failed else ("" if correct else "correctness gate failed")
        _write(a.results_dir, metrics, correct, err)
        return 0 if correct else 2
    except Exception as exc:
        _write(a.results_dir, {"combined_score": 0.0}, False,
               f"{exc}\n{traceback.format_exc()}")
        return 1
    finally:
        if os.path.exists(backup):
            shutil.copy2(backup, dest); os.remove(backup)
        if not a.no_build:
            for bt in {t.build_target for t in tlist}:
                so_rel = next(t.so_path for t in tlist if t.build_target == bt)
                try:
                    build.restore(so_rel)
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
