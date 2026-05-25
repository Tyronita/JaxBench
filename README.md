# JaxBench

**A dataset of optimization challenges on JAX's own hot paths — KernelBench, but the
thing you optimize is JAX itself.**

Each challenge hands you the **current C++/CUDA source** of a JAX hot path as the
reference. You submit a **faster implementation**. JaxBench **rebuilds JAX** with your
change and scores the **speedup of the real library**, gated on **correctness against
stock JAX**. The mission: make the official `jax-ml/jax` measurably faster and land
high-impact upstream PRs.

> The challenge is not "how fast is JAX" — it's "**change a solver / a graph pass in
> C++, rebuild, keep it correct, make it faster.**" Correctness is a gate; speedup is
> the score.

## Two tiers of hot path

| tier | what you edit | rebuild | why it matters |
|---|---|--:|---|
| **`jaxlib_kernel`** | leaf C++/CUDA kernels — LU/QR/SVD/eigh/Cholesky solvers, Threefry PRNG, sparse, tridiagonal | ~4–23s (targeted `.so` + hot-swap) | the library-call path (`jnp.linalg.*`, `jax.random`, sparse) |
| **`xla_core`** | XLA compiler-graph logic — **scan/while lowering, algebraic simplifier, fusion, HLO IR** | ~27–134s (`--override_repository`) | the **computational** wins: faster `jax.lax.scan`, expensive ops, fusion → makes a whole class of programs faster |

Including XLA core is the point: optimizing `scan_expander`, `while_loop_simplifier`,
the `algebraic_simplifier`, and fusion is how you make JAX *more effective*, not just
how you speed up one library call. Their higher rebuild cost is exactly why the loop
runs **serverless + parallel + cross-device** (below).

Full dataset (15 challenges, 10 kernel + 5 XLA core): [`docs/CHALLENGES.md`](docs/CHALLENGES.md).

## Anatomy of a challenge

Each entry in `jaxbench/challenges.py` is `(file, quick build instructions, config,
speed-up test)`. Two examples:

**`prng_threefry`** (jaxlib_kernel — the cheapest, hottest first challenge)
- **file:** `jaxlib/gpu/prng_kernels.cu.cc` (edit inside `// EVOLVE-BLOCK`)
- **build:** `bazel build … --config=cuda_libraries_from_stubs --config=build_cuda_with_clang //jaxlib/cuda:_prng` (~4.8s)
- **apply:** hot-swap `jax_cuda12_plugin/_prng.so`
- **speed-up test:** `jax.random.uniform` over 2¹⁸…2²⁴; correctness = determinism + distribution

**`xla_scan_expander`** (xla_core — the "scan expensive operations" target)
- **file:** `xla/service/scan_expander.cc` (the scan→while lowering)
- **build:** `bazel build … --override_repository=xla=/data/xla-local //jaxlib/tools:jax_cuda12_plugin_wheel` (~60s)
- **apply:** rebuild + reinstall the plugin wheel (serverless: fresh container per candidate)
- **speed-up test:** `jax.lax.scan` cumulative recurrence over 1k…64k steps; correctness = output matches reference

## The loop (correctness-gated speedup)

```
candidate C++  ──▶  build (tier-aware)  ──▶  apply (.so hot-swap | wheel reinstall)
                                              │
   score = mean speedup vs stock  ◀──  speed-up test  ──▶  correctness vs stock JAX
   (0 if any test fails correctness)         (jax.lax.scan / jnp.linalg.* / …)
```

`python -m jaxbench.challenge_runner <challenge_id> --device gpu --build`

## Serverless, cross-device

A candidate evaluation is a pure function, so it fans out one-invocation-per-candidate
across **GPU, TPU, and CPU**. Mount a warm bazel `--disk_cache` read-only so the cold
build is paid once globally; each candidate then does only the targeted rebuild
(seconds for kernels) or an XLA override build (parallelised). `docker/Dockerfile`
(`DEVICE=cpu|cuda|tpu`), `serverless/` (handler + Modal app + GCP-TPU path), and
`infra/` (Azure CPU VM + GCP TPU VM) make this one command per device.
Details + recommendation: [`serverless/README.md`](serverless/README.md).

## Validated on real hardware

- **A100 80GB:** stock baselines recorded for 9 ops (`results/gpu/`); the full
  **mutate→build→hot-swap→run loop validated end-to-end** — a fork-built `_linalg.so`
  hot-swapped into stock `jax[cuda12]` ran `cholesky_update` correctly (residual 9.4e-8).
- **24-core CPU:** stock baselines recorded (`results/cpu/`); challenge runner verified
  for both tiers (kernel + XLA scan/while/simplifier workloads run + check on CPU).
- **TPU:** device class + workloads + commands wired; runs on **GCP** (Azure has no
  TPUs) via `infra/gcp_tpu_vm.sh`. No TPU numbers are claimed until that run executes.

Stock baselines are the references each challenge's speedup is scored against.

## Build-speed methodology (why the rebuild is fast enough to evolve)

Per single-kernel edit: naive `nvcc` wheel ~21s → **targeted `.so` + clang device +
hot-swap ~13s** (~4s host-only). Levers: content-addressed `--disk_cache`, one GPU
arch (`sm_80`), clang device compiler, build one extension `.so`, hot-swap it. XLA-tier
edits use `--override_repository` (the http_archive cache can't be edited in place).
Full detail + measured per-file costs: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Evolving with ShinkaEvolve (real dependency)

```bash
pip install -e ".[evolve]"               # pulls shinka-evolve from GitHub
export ANTHROPIC_API_KEY=...             # ShinkaEvolve's LLM
python shinka/run_evo.py --file jaxlib/gpu/prng_kernels.cu.cc --generations 20
```

ShinkaEvolve mutates the hot path's region; `shinka/evaluate.py` builds → applies →
correctness-gates → scores speedup. One island per hot path. See
[`shinka/README.md`](shinka/README.md) and [`docs/PR_PLAYBOOK.md`](docs/PR_PLAYBOOK.md).

## Quickstart

```bash
pip install -e . && pip install "jax[cpu]"     # or jax[cuda12] / jax[tpu]
python -m jaxbench.challenges                   # the challenge dataset
python -m jaxbench.challenge_runner xla_scan_expander --device cpu   # run one (no build)
```

## Layout

```
jaxbench/  challenges(dataset) · workloads(speed-up tests) · challenge_runner
           build(targeted .so + XLA override) · ops · reference · bench · metrics · sharding
docker/    build+eval image (DEVICE=cpu|cuda|tpu)
serverless/ runtime-agnostic handler · Modal app · cross-device guide
infra/     az (CPU VM) + gcloud (TPU VM) provisioning
shinka/    ShinkaEvolve evaluator + launcher (the evolutionary driver)
docs/      CHALLENGES · METHODOLOGY · METRICS · PR_PLAYBOOK
```

## License

Apache-2.0 (matches JAX), so improvements flow upstream.
