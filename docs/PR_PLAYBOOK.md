# From an evolved kernel to a merged JAX PR

The goal of JaxBench is at least one **high-impact, approved upstream PR** to
`jax-ml/jax`. Speed alone doesn't merge — this is the bar and the path.

## What makes a candidate PR-worthy

1. **Robust speedup**, not a lucky size: faster across the *entire* `N`-sweep **and**
   every dtype of the op, on the reference machine. One-point wins get rejected.
2. **Correctness preserved**: all JaxBench residuals green, plus the upstream
   `tests/linalg_test.py` (or the op's test) passes unchanged.
3. **No API/behaviour change**: same outputs, dtypes, shapes, error semantics.
4. **Self-contained**: the change lives in the kernel `.cc`/`.cu.cc`; no new deps, no
   XLA-core churn.
5. **Portable or clearly scoped**: ideally helps across archs; if `sm_80`-specific,
   say so and guard it.

## Pick the target with leverage

Cross **runtime hotness** with **maintainability** (see `TARGETS_TOP100.md` /
hotpath ranking). Best bets: a widely-used op where the current kernel is simple
enough to improve and the change is reviewable — e.g. `solver_kernels_ffi.cc`
(cuSOLVER LU/QR/eigh, most-maintained GPU file) or the proven
`tridiagonal_solve_perturbed` path. Avoid first PRs in 22.7 s monoliths or shared
headers; if the win is there, **refactor the hot region into its own small `.cc`** as
a separate, reviewable prep PR.

## The path

1. **Evolve** under JaxBench until a candidate clears the bar above.
2. **Confirm out-of-loop**: run the full upstream op test suite; profile with
   `ncu` to show *why* it's faster (occupancy / bandwidth / fewer stalls) — reviewers
   want the mechanism, not just a number.
3. **Benchmark table** for the PR: baseline vs candidate latency + speedup across the
   `N`-sweep and dtypes, with the exact machine + CUDA/toolchain versions.
4. **Minimal diff**: keep the `EVOLVE-BLOCK` change tight and human-readable; add a
   comment explaining the optimisation.
5. **Open the PR** against `jax-ml/jax` referencing the JaxBench task id(s),
   the reproduction command, and the results JSON. CC the linalg/GPU owners.
6. **Cross-platform**: confirm CPU path unaffected and, where relevant, that the
   kernel still builds/works for other archs.

## Reproducibility for reviewers

Every claim must reproduce from:

```bash
python shinka/make_baseline.py --device gpu          # baseline on their box
python -m jaxbench.runner <task_id> --build          # build+swap+correctness+perf
```

A PR that a reviewer can reproduce in two commands, that shows a mechanism and a
robust speedup with tests green, is the one that merges.
