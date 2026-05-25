# JaxBench

**A KernelBench-style benchmark for *evolving* JAX/`jaxlib` kernels.**

JaxBench defines 100 tasks over the real linear-algebra, PRNG and sparse kernels in
`jaxlib`. For each task it pins the JAX operation, the C++/CUDA source file that backs
it (the mutation surface), the *targeted* build + `.so` hot-swap recipe, and the
correctness + speedup evaluation. [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)
mutates the kernel; JaxBench builds it in seconds, gates on correctness, and scores
the speedup. The aim: at least one **high-impact, approved PR to upstream JAX**.

> Like KernelBench, **correctness is a gate and speedup is the score** — a
> fast-but-wrong kernel scores zero. Unlike KernelBench, the hard part here is making
> the *rebuild* fast enough to evolve: we get a single GPU kernel edit down to **~13 s**
> (vs ~21 s for a naive wheel rebuild). See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Why

The bottleneck of an evolutionary kernel loop (mutate → rebuild → benchmark) is the
**rebuild**. JaxBench bakes in the measured-fastest recipe — content-addressed
disk cache, single GPU arch, **clang as the CUDA device compiler**, rebuild just the
one affected extension `.so`, and **hot-swap** it instead of repackaging a wheel.

## Layout

```
jaxbench/
  registry.py     # 100 tasks: op × device × dtype → file, build target, .so, N-sweep
  ops.py          # input generation + the JAX op under test (dispatches to jaxlib)
  reference.py    # property-based correctness residuals (robust to non-uniqueness)
  correctness.py  # the correctness gate
  bench.py        # latency / GFLOP/s with proper JAX timing hygiene
  metrics.py      # FLOP models, speedup, metric definitions
  build.py        # targeted .so rebuild (clang device) + hot-swap into the venv
  runner.py       # build → swap → correctness → perf → speedup → JSON
  sharding.py     # multi-GPU sharded-batch evaluation
  tasks.yaml      # the registry, serialised
tests/            # parameterised correctness (all 100), integration, build smoke
shinka/           # ShinkaEvolve evaluator, island config, baseline recorder
docs/             # methodology, top-100 targets, metrics, PR playbook
```

## Quickstart

```bash
pip install -e .                      # numpy, scipy, pyyaml (+ jax for running)

# inspect the 100 tasks
python -m jaxbench.registry

# correctness on CPU (GPU tasks auto-skip without a GPU)
JAX_ENABLE_X64=1 pytest tests/ -q

# record a baseline, then run one task end-to-end (build + hot-swap + eval)
python shinka/make_baseline.py --device gpu
python -m jaxbench.runner cholesky_update__gpu__f32 --build
```

## The 100 tasks

Generated from `jaxbench/registry.py` — full table in
[`docs/TARGETS_TOP100.md`](docs/TARGETS_TOP100.md). Families: dense linalg
(LU/QR/SVD/eigh/solve/inv/cholesky/det/expm/lstsq), Threefry PRNG, sparse CSR matmul,
tridiagonal solves; on GPU (`jax-cuda-plugin`) and CPU (`jaxlib`), across
`f32/f64/c64/c128`.

## Evolving with ShinkaEvolve

One island per mutation-surface file, ordered cheapest-rebuild-first. See
[`shinka/README.md`](shinka/README.md). Fitness = mean speedup across the file's
tasks, `0` if correctness fails.

## Status & roadmap

- ✅ 100-task registry, correctness gate (validated on CPU), perf harness, targeted
  build + hot-swap, ShinkaEvolve adapter, multi-GPU sharding eval, full docs.
- ⏭️ Record GPU baselines on the reference A100; evolve the high-leverage islands;
  land the first upstream PR (see [`docs/PR_PLAYBOOK.md`](docs/PR_PLAYBOOK.md)).

## License

Apache-2.0 (matches JAX), so improvements can flow upstream.
