# Driving JaxBench with ShinkaEvolve

ShinkaEvolve is a **real dependency** here (`pip install -e ".[evolve]"` pulls
`shinka-evolve` from GitHub). It is the outer evolutionary loop; JaxBench is the
evaluator.

```
ShinkaEvolve mutates code inside // EVOLVE-BLOCK-START/END   (LLM-guided)
        │   writes candidate kernel -> program_path
        ▼
shinka/evaluate.py --program_path <candidate> --results_dir <dir>
        │  stage candidate into the jax checkout
        │  build the one affected extension .so   (clang device, ~5–13s)
        │  hot-swap the .so into the venv
        │  correctness gate ── fail ─▶ combined_score = 0.0
        │  latency sweep → mean speedup vs baseline
        ▼
results_dir/metrics.json {combined_score, ...} + correct.json {correct, error}
        │
        ▼  ShinkaEvolve maximises combined_score
```

This matches ShinkaEvolve's eval contract exactly (`--program_path` / `--results_dir`,
`metrics.json` with `combined_score`, `correct.json`).

## Run it

```bash
pip install -e ".[evolve]"               # installs shinka-evolve (+ jaxbench)
export ANTHROPIC_API_KEY=...             # or OPENAI_API_KEY — ShinkaEvolve's LLM
python shinka/make_baseline.py --device gpu     # 1) record baselines once
python shinka/run_evo.py \                       # 2) evolve one island
    --file jaxlib/gpu/prng_kernels.cu.cc --generations 20
```

`run_evo.py` builds a ShinkaEvolve `EvolutionConfig` + `ShinkaEvolveRunner` with
`init_program_path` = the live kernel file and `eval_program_path` = `evaluate.py`,
sets `SHINKA_TARGET_FILE` so the evaluator knows which kernel/tasks to score, and
calls `runner.run()`.

## Islands

`shinka_config.yaml` lists one island per mutation-surface file, ordered
cheapest-rebuild-first so the highest-value, fastest-to-iterate kernels evolve first:

| order | file | rebuild | why |
|--:|---|--:|---|
| 1 | `gpu/prng_kernels.cu.cc` | ~4.8 s | hottest path, cheapest loop |
| 2 | `gpu/linalg_kernels.cc` | ~4.2 s | host glue |
| 3 | `gpu/solver_kernels_ffi.cc` | ~7 s | cuSOLVER LU/QR/eigh, most-maintained |
| 4 | `gpu/householder_kernels.cu.cc` | ~5.5 s | QR building block |
| 5 | `gpu/linalg_kernels.cu.cc` | ~12.9 s | heavy device TU |
| 6 | `tridiagonal_solve_perturbed.h` | 6.6–17 s | proven +41.7% (refactor header first) |
| 7 | `cpu/lapack_kernels.cc` | 22.7 s | split before evolving |

## Fitness contract

- **Correctness is a hard gate**: any task for the file failing its residual ⇒
  `combined_score = 0.0`.
- Otherwise `combined_score` = **mean speedup** across the file's tasks over the full
  N-sweep vs the recorded baseline.

## Why it's fast

The evaluator never rebuilds the wheel — it rebuilds only the affected extension `.so`
with clang as the CUDA device compiler and hot-swaps it. See
[`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md). For scale, run each candidate as a
serverless invocation — [`../serverless/README.md`](../serverless/README.md).
