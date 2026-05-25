# JaxBench — the plan

*KernelBench, but for **evolving** JAX/`jaxlib` kernels — and engineered so the
rebuild is fast enough that an evolutionary loop is actually practical.*

**Repo:** https://github.com/Tyronita/JaxBench ·
**Fork test branch:** `Tyronita/jax @ shinka/jaxbench-tests` ·
**Driver:** [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) ·
**Goal:** ≥ 1 high-impact, approved PR to upstream `jax-ml/jax`.

---

## The idea

KernelBench judges generated CUDA kernels by *correctness first, speedup second*.
JaxBench applies that to the real kernels inside `jaxlib` (dense linear algebra,
Threefry PRNG, sparse, tridiagonal), and adds the missing piece for **evolution**: a
mutate→build→measure loop whose rebuild is seconds, not minutes.

> Correctness is a **gate** (a fast-but-wrong kernel scores 0). Speedup over the
> unmutated baseline is the **score**. ShinkaEvolve maximises it.

## Why it's hard — and how we made it fast

The bottleneck of evolving a compiled kernel is the **rebuild**. We measured every
lever on an A100 / hermetic CUDA 12.9 box and baked the winners into the loop:

| approach | per-edit rebuild |
|---|--:|
| naive full `nvcc` wheel | ~21 s |
| targeted extension `.so`, `nvcc` | ~17 s |
| **targeted `.so`, clang device compiler + `.so` hot-swap** | **~13 s** (device) / ~4 s (host) |

Recipe: content-addressed `--disk_cache` on a big disk · build **one GPU arch**
(`sm_80`) · **clang** as the CUDA device compiler (−25–36 %) · rebuild only the one
affected extension · **hot-swap** the `.so` into the venv instead of repackaging a
wheel · `-c opt` (fastbuild is *not* faster for device code). And the hard rule:
**never mutate XLA core graph code** — editing `xla/hlo/ir/hlo_instruction.h` triggers
a 134 s / 237-action rebuild (fans out to 98 libraries); mainstream DL ops are
XLA-codegen'd anyway. Keep mutations in leaf `jaxlib` kernels.

## What's in the benchmark

- **100 tasks** = op × device × dtype, generated from `jaxbench/registry.py`, each
  pinning the JAX op, the backing `jaxlib` file (the mutation surface), the build
  target, the extension `.so`, and the N-sweep. Full table: `docs/TARGETS_TOP100.md`.
  Families: LU/QR/SVD/eigh/solve/inv/cholesky/det/expm/lstsq, Threefry PRNG, CSR
  sparse, tridiagonal — on GPU (`jax-cuda-plugin`) and CPU (`jaxlib`), f32/f64/c64/c128.
- **Correctness gate**: property residuals (`QR≈A`, `A·x≈b`, `L·Lᴴ≈A`, …) robust to
  factorisation non-uniqueness; dtype-scaled tolerances; determinism checks.
  Parameterised over all 100 tasks — **validated green on CPU** (GPU tasks auto-skip
  without a GPU).
- **Perf harness** with correct JAX timing hygiene (warmup excluded, inputs staged
  once, `block_until_ready`, median of 20), op-specific GFLOP/s, speedup.
- **Build + hot-swap** module implementing the fast loop; **multi-GPU sharding** eval
  (`jaxbench/sharding.py`) to confirm wins hold under data-parallel batches.
- **ShinkaEvolve adapter** (`shinka/`): one island per file, ordered cheapest-rebuild
  first; fitness = mean speedup, 0 if correctness fails; baseline recorder.

## Methodology highlights

- **Target machines** are first-class: each task carries a device class; switching to
  H100/T4 means changing the compute capability and re-recording the baseline.
- **Side-effect-free compiles**: hermetic + content-addressed builds; the only
  environment mutation is one atomic `.so` swap (reverted by `restore()`); correctness
  runs in a fresh process. → safe to fan out.
- **Serverless execution**: seed ephemeral GPU workers from a warm `--disk_cache`
  snapshot (read-only mount); each worker rebuilds only its mutated extension
  (seconds), evaluates its task subset in an isolated venv, and writes append-only
  fitness JSON — one mutation per invocation, nothing shared mutated.
- **Other GPU tooling**: Nsight/`ncu` for occupancy/bandwidth confirmation on
  promising candidates (out of the hot loop); CUPTI/`nvidia-smi dmon` for clock/power
  sanity during timing.

## The path to an upstream PR (`docs/PR_PLAYBOOK.md`)

1. Evolve the high-leverage islands (`prng`, `solver`, the proven tridiagonal path).
2. Promote only candidates with a **robust speedup across the whole N-sweep and all
   dtypes**, all JaxBench + upstream op tests green.
3. Confirm out-of-loop with `ncu` (show the *mechanism*), build a benchmark table with
   exact machine/toolchain versions.
4. Open a minimal, reproducible PR against `jax-ml/jax` referencing the JaxBench task
   ids and a two-command repro.

## Status

✅ Repo published · 100-task registry · correctness gate (CPU-validated) · perf +
sharding harness · build/hot-swap loop · ShinkaEvolve adapter · fork test branch ·
CI · full docs.
⏭️ Record GPU baselines on the A100 → evolve → land the first PR.

## Reproduce

```bash
git clone https://github.com/Tyronita/JaxBench && cd JaxBench
pip install -e . && pip install "jax[cpu]"
JAX_ENABLE_X64=1 pytest tests/ -q            # correctness on CPU
python -m jaxbench.registry                  # the 100 tasks
# on a GPU box:
python shinka/make_baseline.py --device gpu
python -m jaxbench.runner threefry_uniform__gpu__f32 --build
```
