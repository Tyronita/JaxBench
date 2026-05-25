#!/usr/bin/env python3
"""Launch ShinkaEvolve on a JaxBench island (real ShinkaEvolve dependency).

Requires `pip install -e ".[evolve]"` (pulls shinka-evolve from GitHub) and an LLM key
(e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY) per ShinkaEvolve's docs.

    python shinka/run_evo.py --file jaxlib/gpu/prng_kernels.cu.cc --generations 20

ShinkaEvolve mutates the kernel's EVOLVE-BLOCK region and scores each candidate via
shinka/evaluate.py (build -> hot-swap -> correctness gate -> speedup). Islands and
their rebuild cost are listed in shinka_config.yaml.
"""
from __future__ import annotations
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from jaxbench import build, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="jaxlib kernel file (registry key) to evolve")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=1, help="parallel candidate evals")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "evo"))
    a = ap.parse_args()

    if not any(t.file == a.file for t in registry.tasks()):
        raise SystemExit(f"{a.file} is not a JaxBench mutation surface; see docs/TARGETS_TOP100.md")

    # the initial program ShinkaEvolve evolves is the live kernel file in the jax checkout
    init_program = os.path.join(build.JAX_REPO, a.file)
    if not os.path.exists(init_program):
        raise SystemExit(f"kernel not found: {init_program} (set JAXBENCH_JAX_REPO)")

    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    # evaluate.py needs to know which kernel it's scoring
    os.environ["SHINKA_TARGET_FILE"] = a.file

    job = LocalJobConfig(eval_program_path=os.path.join(HERE, "evaluate.py"))
    db = DatabaseConfig(db_path=os.path.join(a.results, "shinka.sqlite"))
    evo = EvolutionConfig(
        num_generations=a.generations,
        max_parallel_jobs=a.parallel,
        init_program_path=init_program,
        language="cpp",
        results_dir=a.results,
        task_sys_msg=(
            "You are optimising a JAX/jaxlib GPU/CPU kernel for speed. Only modify code "
            "between // EVOLVE-BLOCK-START and // EVOLVE-BLOCK-END. Preserve the function "
            "signatures, numerical results (correctness is gated), and build-ability. "
            "Prefer changes that reduce latency without changing outputs."
        ),
    )
    ShinkaEvolveRunner(evo_config=evo, job_config=job, db_config=db).run()


if __name__ == "__main__":
    main()
