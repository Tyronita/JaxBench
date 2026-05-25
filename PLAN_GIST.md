# JaxBench — the plan

**A dataset of optimization challenges on JAX's own hot paths. KernelBench, but the
thing you optimize is JAX itself: change a solver or a graph pass in C++, rebuild JAX,
keep it correct, make it faster.**

**Repo:** https://github.com/Tyronita/JaxBench ·
**Driver:** [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (real dependency) ·
**Goal:** ≥ 1 high-impact, approved PR to upstream `jax-ml/jax`.

---

## What it is (and what it is *not*)

It is **not** a benchmark of how fast JAX is. It is a **dataset of challenges**. Each
challenge gives you the **current C++/CUDA source** of a JAX hot path as the reference;
you submit a **faster implementation**; JaxBench **rebuilds JAX** and scores the
**speedup of the real library**, gated on **correctness against stock JAX**.

A challenge is `(file, quick build instructions, config, speed-up test)`:
- **file** — the editable hot path (the reference impl lives here).
- **build** — the exact targeted rebuild command + config that makes it fast/correct.
- **speed-up test** — a JAX workload that routes through this hot path.
- **score** — mean speedup vs stock over the test sweep; **0 if any test breaks correctness**.

## Two tiers of hot path

- **`jaxlib_kernel`** — leaf C++/CUDA kernels: LU/QR/SVD/eigh/Cholesky solvers,
  Threefry PRNG, sparse, tridiagonal. Cheap to rebuild (~4–23 s) via a targeted
  extension-`.so` build + hot-swap.
- **`xla_core`** — XLA compiler-graph logic: **scan/while lowering, the algebraic
  simplifier, instruction fusion, HLO IR**. This is where the *computational* wins are
  — making `jax.lax.scan`, expensive ops, and fusion genuinely faster makes a whole
  class of programs faster. Costlier to rebuild (~27–134 s, via
  `--override_repository`), which is exactly why the loop runs serverless + parallel.

(15 challenges today: 10 kernel + 5 XLA core. The `scan_expander` challenge is the
"scan expensive operations" target directly.)

## The loop

```
candidate C++ ─▶ build (tier-aware) ─▶ apply (.so hot-swap | wheel reinstall)
                                          │
  score = mean speedup vs stock ◀─ speed-up test ─▶ correctness vs stock JAX
  (0 if incorrect)                  (jax.lax.scan / jnp.linalg.* / fusible graph / …)
```

## Serverless, cross-device

A candidate eval is a pure function, so it fans out one-invocation-per-candidate across
**GPU, TPU, and CPU**. Mount a warm bazel `--disk_cache` read-only so the cold build is
paid once globally; each candidate then does only the targeted rebuild (kernels) or an
XLA override build (parallelised). `docker/Dockerfile` (`DEVICE=cpu|cuda|tpu`),
`serverless/` (runtime-agnostic handler + Modal app + GCP-TPU path), and `infra/`
(Azure CPU VM + GCP TPU VM) make each device one command. TPUs are GCP-only.

## Why the rebuild is fast enough to evolve

Per single-kernel edit: naive `nvcc` wheel ~21 s → **targeted `.so` + clang device +
hot-swap ~13 s** (~4 s host-only). Levers: content-addressed `--disk_cache`, one GPU
arch (`sm_80`), clang as the CUDA device compiler, build one extension `.so`, hot-swap
it. XLA-tier edits use `--override_repository` (the http_archive cache can't be edited
in place).

## Validated on real hardware

- **A100 80GB:** stock baselines recorded; the full **mutate→build→hot-swap→run loop
  validated end-to-end** — a fork-built `_linalg.so` hot-swapped into stock
  `jax[cuda12]` ran `cholesky_update` correctly (residual 9.4e-8).
- **24-core CPU:** stock baselines recorded; challenge runner verified for both tiers
  (kernel + XLA scan/while/simplifier workloads run + correctness-check on CPU).
- **TPU:** wired for GCP (Azure has no TPUs); no TPU numbers claimed until run there.

## Path to an upstream PR

Evolve a challenge until a candidate shows a **robust speedup across the whole sweep**
with all correctness green and the upstream op/graph tests passing; confirm the
mechanism with `ncu`; open a minimal, reproducible PR to `jax-ml/jax`. The leverage
targets: hot kernels with simple, reviewable bodies (`solver`, `prng`, the proven
tridiagonal path) and the high-impact graph passes (`scan`, fusion).

## Reproduce

```bash
git clone https://github.com/Tyronita/JaxBench && cd JaxBench
pip install -e . && pip install "jax[cpu]"
python -m jaxbench.challenges                                  # the dataset
python -m jaxbench.challenge_runner xla_scan_expander --device cpu   # run one
```
