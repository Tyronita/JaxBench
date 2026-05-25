# JaxBench evaluation metrics

A candidate kernel is judged like KernelBench: **correctness is a gate, speedup is
the score.**

## Gate: correctness

For each task we run the JAX op across its `N`-sweep and compute a **property
residual** against a trusted reference (not raw factor equality, which is ill-defined
for decompositions):

| op class | residual checked |
|---|---|
| LU / QR / SVD | `‖P·L·U − A‖`, `‖Q·R − A‖`, `‖U·Σ·Vᴴ − A‖` |
| eigh | `‖A·V − V·Λ‖` and orthonormal `V` |
| cholesky / chol-update | `‖L·Lᴴ − A‖` |
| solve / lu_solve / lstsq | `‖A·x − b‖` |
| inv | `‖A·A⁻¹ − I‖` |
| det / slogdet | vs numpy reference (well-scaled inputs) |
| tridiagonal | reconstruct dense system, `‖A·x − b‖` |
| householder | orthonormal columns `‖QᴴQ − I‖` |
| sparse matmul | vs dense `A·x` |
| PRNG | statistical (mean/var) + determinism |

Tolerance is dtype-scaled: `f32`/`c64` → 2e-4, `f64`/`c128` → 1e-10. **Any** task
over tolerance ⇒ the candidate's fitness is `0`.

## Score: speedup

For correct candidates:

- **latency_ms** — median wall time per call, warmup excluded, device-synchronised.
- **throughput_gflops** — op-specific FLOPs ÷ latency (machine-comparable). FLOP
  models live in `jaxbench/metrics.py` (e.g. LU `⅔n³`, SVD `22n³`, eigh `9n³`;
  complex dtypes ×4).
- **speedup_vs_baseline** — `baseline_latency / candidate_latency`, per `N`.
- **fitness** — mean speedup over the file's tasks and the full `N`-sweep
  (`0` if incorrect). This is what ShinkaEvolve maximises.

## Reporting

Each run writes `results/result_<task>.json` with build time, correctness residuals,
the per-size latency/GFLOP/s sweep, and per-size speedup. A candidate is "high
impact" when it shows a **robust speedup across the whole N-sweep and all dtypes** of
an op (not just one favourable size), which is also the bar for an upstream PR — see
[`PR_PLAYBOOK.md`](PR_PLAYBOOK.md).
