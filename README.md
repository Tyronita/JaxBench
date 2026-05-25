# JaxBench

**These are 100 benchmark tasks built on the real C++/CUDA kernels inside JAX's
`jaxlib`** — LU, QR, SVD, eigendecomposition, Cholesky, linear solves, matrix inverse,
determinant, matrix exponential, least-squares, Threefry PRNG, sparse CSR mat-mul and
tridiagonal solves. **Each task pins (a) the JAX operation a user actually calls,
(b) the exact kernel source file that computes it, (c) how to rebuild just that kernel
in seconds, and (d) how to check it's still correct and measure how much faster it
got.**

**The point: make the official JAX library itself faster.** We evolve these kernels
with [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (a real dependency here),
gate every candidate on correctness, score it by speedup, and aim to land **at least
one high-impact, approved pull request to upstream `jax-ml/jax`**.

> Like **KernelBench**, correctness is a **gate** and speedup is the **score** — a
> fast-but-wrong kernel scores zero. The hard part we solved is making the *rebuild*
> fast enough to evolve: one GPU kernel edit rebuilds in **~13 s** (vs ~21 s for a
> naive wheel) via a clang-device targeted build + `.so` hot-swap.

## Three device classes

JaxBench is defined across **three device classes — NVIDIA GPU (A100), TPU, and
multi-core CPU** — because the same `jnp.linalg.*` call runs on all three JAX
backends, and a kernel win should be checked on each.

| device | how it's run | status in this repo |
|---|---|---|
| **A100 GPU** (`sm_80`) | A100 80GB + `jax[cuda12]`; targeted clang build + `.so` hot-swap | **real results recorded** (`results/gpu/`) |
| **CPU** (24-core) | `jax[cpu]`, runs anywhere | **real results recorded** (`results/cpu/`) |
| **TPU** | GCP TPU VM / Cloud Run (TPUs are Google-Cloud-only) | device class + tasks + commands wired; run via `infra/gcp_tpu_vm.sh` |

> Honesty note: GPU and CPU numbers below are **measured** (A100 80GB and the 24-core
> host). **TPU runs on GCP** — Azure has no TPUs — so the TPU device class is fully
> wired and one command away, but no TPU numbers are claimed until that run is
> executed. See `infra/README.md`.

### Real A100 80GB results (`f32`, latency ms, all correctness-gated ✅ 9/9)

| task | N=64 | N=256 | N=1024 |
|---|--:|--:|--:|
| `cholesky__gpu__f32` | 0.378 | 0.812 | 0.908 |
| `lu__gpu__f32` | 0.856 | 0.971 | 4.600 |
| `solve__gpu__f32` | 0.841 | 1.238 | 4.934 |
| `qr__gpu__f32` | 0.747 | 1.936 | 9.536 |
| `cholesky_update__gpu__f32` | 1.296 | 2.326 | 11.141 |
| `eigh__gpu__f32` | 1.683 | 4.138 | 18.911 |
| `svd__gpu__f32` | 8.072 | 19.809 | 126.280 |

### Real CPU results (24-core x86_64, `f64`, latency ms)

| task | N=64 | N=256 | N=1024 |
|---|--:|--:|--:|
| `lu__cpu__f64` | 0.077 | 0.826 | 924.40 |
| `qr__cpu__f64` | 0.179 | 307.48 | 2337.21 |
| `svd__cpu__f64` | 43.01 | 450.46 | 5204.47 |
| `cholesky__cpu__f64` | 0.038 | 0.551 | 11.34 |
| `solve__cpu__f64` | 0.072 | 0.700 | 918.74 |
| `tridiagonal_solve__cpu__f64` | 0.000 | 0.016 | 0.05 |

Reproduce: `JAX_ENABLE_X64=1 python -m jaxbench.runner lu__cpu__f64` (or any task id).

## Anatomy of a task — what each part represents

A task is one row of `jaxbench/registry.py`. Three worked examples:

### 1. `cholesky_update__gpu__f32` — a GPU device kernel

| field | value | what it means |
|---|---|---|
| `id` | `cholesky_update__gpu__f32` | unique name = `op__device__dtype` |
| `op` | `cholesky_update` | the JAX call under test: `jax.lax.linalg.cholesky_update` |
| `file` | `jaxlib/gpu/linalg_kernels.cu.cc` | **mutation surface** — the CUDA source ShinkaEvolve edits |
| `device` / `dtype` | `gpu` / `f32` | run on the A100; single-precision |
| `build_target` | `//jaxlib/cuda:_linalg` | the one Bazel target rebuilt per edit |
| `so_path` | `jax_cuda12_plugin/_linalg.so` | the extension `.so` hot-swapped into the venv |
| `wheel` | `jax-cuda-plugin` | which wheel this extension belongs to |
| `sizes` | `64…1024` | matrix-dimension sweep used for correctness + perf |
| `tol` | `2e-4` | f32 residual tolerance for the correctness gate |
| `cuda_compiler` | `clang` | device `.cu.cc` ⇒ compiled with clang (25–36% faster rebuild) |
| `metrics` | latency, GFLOP/s, speedup, correctness | what's recorded |

### 2. `lu__cpu__f64` — a CPU LAPACK kernel

| field | value | what it means |
|---|---|---|
| `op` | `lu` | `jax.scipy.linalg.lu` (LU via `getrf`) |
| `file` | `jaxlib/gpu/solver_kernels_ffi.cc` (GPU) / `_lapack` on CPU | backend; here built as `//jaxlib/cpu:_lapack` |
| `device` / `dtype` | `cpu` / `f64` | double precision, `tol = 1e-10` |
| `build_target` | `//jaxlib/cpu:_lapack` | rebuilds the CPU LAPACK extension (`jaxlib` wheel) |
| correctness | residual `‖P·L·U − A‖ / ‖A‖ ≤ tol` | property check, robust to factor sign/order |

### 3. `threefry_uniform__gpu__f32` — the PRNG hot path

| field | value | what it means |
|---|---|---|
| `op` | `threefry_uniform` | `jax.random.uniform` (Threefry2x32) |
| `file` | `jaxlib/gpu/prng_kernels.cu.cc` | the cheapest kernel to rebuild (~4.8 s) ⇒ ideal first island |
| `sizes` | `2^16 … 2^24` | vector-length sweep (PRNG is 1-D) |
| correctness | determinism (same key ⇒ same bits) + uniform mean/var | a PRNG mutation must stay deterministic |

Full list of all 100: [`docs/TARGETS_TOP100.md`](docs/TARGETS_TOP100.md).

## The fast rebuild loop (why evolution is practical)

Measured on the A100 box, per single-kernel edit: naive `nvcc` wheel ~21 s →
**targeted `.so` + clang device + hot-swap ~13 s** (≈4 s for host-only `.cc`). Levers:
content-addressed `--disk_cache`, one GPU arch (`sm_80`), clang device compiler, build
one extension `.so`, hot-swap it (the plugin is 11 independent `.so`s). Never mutate
XLA core graph code (134 s / 98-library fan-out). Full detail:
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

> **Validated end-to-end on the A100:** a fork-built `_linalg.so` hot-swapped into a
> stock `jax[cuda12]` install loaded and ran `cholesky_update` correctly (residual
> 9.4e-8) — i.e. the whole mutate→build→hot-swap→run loop works on real hardware.

## Evolving with ShinkaEvolve (real dependency)

```bash
pip install -e ".[evolve]"     # pulls shinka-evolve from GitHub
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY (ShinkaEvolve's LLM)
python shinka/make_baseline.py --device gpu          # record baselines once
python shinka/run_evo.py --file jaxlib/gpu/prng_kernels.cu.cc --generations 20
```

`shinka/run_evo.py` uses ShinkaEvolve's `EvolutionConfig`/`ShinkaEvolveRunner`;
`shinka/evaluate.py` is the eval program it calls (build → hot-swap → correctness gate
→ speedup, writing `metrics.json`/`correct.json`). One island per kernel file, ordered
cheapest-rebuild-first (`shinka/shinka_config.yaml`). See [`shinka/README.md`](shinka/README.md).

## Serverless & Docker

A candidate eval is a pure function, so it fans out one-invocation-per-candidate.
`docker/Dockerfile` builds device images (`DEVICE=cpu|cuda|tpu`); `serverless/`
has the runtime-agnostic handler + a Modal app (GPU+CPU) and the GCP path for TPU;
mount a warm `--disk_cache` read-only so each invocation only does the seconds-long
targeted rebuild. Recommendation + deployment: [`serverless/README.md`](serverless/README.md).

## Quickstart

```bash
pip install -e .                       # numpy, scipy, pyyaml
pip install "jax[cpu]"                 # or jax[cuda12] / jax[tpu] per device
python -m jaxbench.registry            # the 100 tasks
JAX_ENABLE_X64=1 pytest tests/ -q      # correctness (GPU tasks auto-skip on CPU)
JAX_ENABLE_X64=1 python -m jaxbench.runner lu__cpu__f64        # one task, real
```

## Layout

```
jaxbench/   registry(100 tasks) · ops · reference · correctness · bench · metrics · build · runner · sharding
tests/      parameterised correctness (all 100) · integration · build smoke
shinka/     ShinkaEvolve evaluator + launcher + island config + baseline recorder
docker/     parameterised build+eval image (cpu|cuda|tpu)
serverless/ runtime-agnostic handler · Modal app · deployment guide
infra/      az (CPU VM) + gcloud (TPU VM) provisioning
docs/       methodology · top-100 targets · metrics · PR playbook
```

## License

Apache-2.0 (matches JAX), so improvements flow upstream.
