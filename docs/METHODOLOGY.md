# JaxBench methodology

JaxBench scores an evolved jaxlib kernel the way **KernelBench** scores a generated
CUDA kernel: a candidate must first be **correct**, and is then ranked by **speedup**
over the unmutated baseline. The novel part is making the *rebuild* fast enough that
an evolutionary loop (thousands of mutate→build→measure cycles) is practical.

## 1. Target machines

A task pins a **device class**, because kernel performance and the right build flags
are device-specific.

| field | reference machine | notes |
|---|---|---|
| GPU | NVIDIA **A100 80 GB** (`sm_80`) | build a single arch; multi-arch fan-out is the #1 nvcc time sink |
| CPU | x86-64, 24 cores | hermetic LLVM-18 host toolchain |
| CUDA | hermetic **12.9**, clang device compiler | `--config=build_cuda_with_clang` |
| Python | 3.12 (hermetic) | 3.10 has no requirements lockfile in this jaxlib rev |

To target other hardware (H100 `sm_90`, T4 `sm_75`, …) change the compute capability
and re-record the baseline; task definitions are unchanged. Results are reported with
the machine descriptor so numbers are comparable only within a device class.

## 2. The fast rebuild loop (the core contribution)

Measured on the reference A100 box, per single-kernel edit:

| approach | per edit | why |
|---|--:|---|
| full `nvcc` wheel rebuild (naive) | ~21 s | recompiles + relinks + repackages the wheel |
| targeted `.so`, `nvcc` | ~17 s | rebuild one extension, skip wheel packaging |
| **targeted `.so`, clang device + hot-swap** | **~13 s** | clang device compile (−25–36%) + copy the `.so` |
| host-only (`.cc`) edit | ~4 s | one host TU recompile + relink |

Levers, in priority order:

1. **`--disk_cache` on a large separate disk** — content-addressed, survives
   `bazel clean` and process kills, shared across configs. Restored 3,773/6,623
   actions instantly after two killed builds.
2. **One GPU arch** (`HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_80`).
3. **clang as the CUDA device compiler** (`--config=build_cuda_with_clang`):
   25–36 % faster `.cu.cc` compiles; host code unaffected. *Commit to it* — flipping
   nvcc↔clang discards Bazel's analysis cache.
4. **Rebuild the one affected extension `.so`**, not the wheel.
5. **Hot-swap the `.so`** into the installed plugin (the plugin is 11 independent
   extension `.so`s, not a monolith). No wheel repackage.
6. `--features=-layering_check` (fixes the hermetic crosstool module clash);
   `-c opt` (fastbuild is *not* faster for device code).

**Never mutate XLA core graph code in the loop.** Measured rebuild after editing
`xla/hlo/ir/hlo_instruction.h` = 134 s / 237 actions (it fans out to 98 libraries);
even a single XLA leaf TU is 27–43 s. And mainstream DL ops (matmul/conv/attention)
are XLA-codegen'd, not these files. Keep the mutation surface in leaf jaxlib kernels.

## 3. Side-effect-free compiles & serverless execution

The loop must be reproducible and isolated so parallel candidates don't corrupt each
other:

- **Hermetic + content-addressed.** The toolchain, CUDA, and Python are hermetic;
  every action is keyed by content and cached in `--disk_cache`. Identical inputs ⇒
  identical artifact ⇒ cache hit. No host-state leakage.
- **The only mutation is one `.so`.** `hot_swap()` backs up the installed extension
  and atomically replaces a single file; `restore()` reverts it. Nothing else in the
  environment changes. Correctness is then evaluated in a **fresh process** (the
  extension is loaded at import), so there is no in-process state carryover.
- **Serverless / parallel fan-out.** Because builds are content-addressed and the
  swap is a one-file copy, candidates parallelise across ephemeral workers:
  1. seed each worker from a warm `--disk_cache` snapshot (object store / shared
     volume) so the cold build is paid once, globally;
  2. each worker rebuilds only its mutated extension (seconds) and runs its task
     subset in an isolated venv/container;
  3. results (fitness JSON) are collected centrally; no worker writes shared state
     beyond the append-only results store.
  This maps cleanly onto serverless GPU runners (one mutation per invocation, warm
  cache mounted read-only, `.so` built to a scratch dir, evaluated, discarded).

## 4. Measurement hygiene

- JIT **warmup excluded**; inputs transferred to device **once** outside the timing
  loop; every call `block_until_ready()`-synced; **median** of 20 trials.
- Correctness uses **property residuals** (e.g. `QR≈A`, `A@x≈b`, `LLᴴ≈A`) so it is
  robust to the sign/order non-uniqueness of decompositions; tolerance scales with
  dtype (`f32` 2e-4, `f64` 1e-10).
- Determinism is checked (same seed ⇒ identical residual) to reject flaky kernels.

## 5. Other GPU tooling (optional, supported)

- **Nsight Compute / `ncu`** for occupancy, achieved bandwidth, warp stalls on a
  promising candidate (profile-guided confirmation, not in the hot loop).
- **CUPTI / `nvidia-smi dmon`** for power and clock-stability sanity during timing.
- **Multi-GPU sharding** (`jaxbench/sharding.py`): batched-matrix tasks are sharded
  across devices with `jax.sharding` to confirm a kernel win holds under data
  parallelism and introduces no host-sync/collective regression.
