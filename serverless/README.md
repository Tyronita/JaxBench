# Serverless execution & Docker for JaxBench

**Short answer: yes — Docker + serverless is the right way to run JaxBench at scale,
and it's exactly what an evolutionary loop wants.** A candidate evaluation is a pure
function (`build → hot-swap → correctness → speedup`), so it maps cleanly onto
one-invocation-per-candidate serverless fan-out. KernelBench uses Docker for the same
reason: reproducible, isolated compilation.

## Why it fits

- **Side-effect-free.** The build is hermetic + content-addressed; the only state a
  worker mutates is one extension `.so` in its own container, reverted after eval.
- **Warm cache amortised globally.** Mount the bazel `--disk_cache` **read-only** from
  a shared volume/object store. The cold jaxlib build is paid **once**; each candidate
  then only does the seconds-long *targeted* rebuild (clang device, ~5–13s).
- **Embarrassingly parallel.** N candidates = N invocations; no shared mutable state
  beyond an append-only results store.

## Architecture

```
ShinkaEvolve (controller)                shared volume / object store
   │  emits candidate {file, source}        ├─ bazel-disk/   (warm cache, RO mount)
   ▼                                         └─ jax/          (checkout, RO/COW)
serverless invocation  ── per candidate ──▶ container (device image)
   handler.handle(event):                       │ targeted .so rebuild (clang)
     build → hot-swap → correctness → perf      │ run in fresh process
   returns metrics {combined_score, ...}  ◀─────┘ revert .so
```

`handler.py` is the runtime-agnostic core; wrap it per platform.

## Per-device deployment

| device class | where serverless works | image | notes |
|---|---|---|---|
| **GPU (A100/H100)** | Modal, RunPod, GKE/Cloud Run GPU, Azure Container Apps GPU | `docker/Dockerfile` `--build-arg DEVICE=cuda` | `--gpus all`; `modal_app.py::evaluate_gpu` ready |
| **CPU (many cores)** | Modal, Cloud Run, AWS Lambda (≤15min)/Fargate, Azure Container Apps | `DEVICE=cpu` | cheapest; `modal_app.py::evaluate_cpu` |
| **TPU** | **GCP only** (Cloud Run TPU / GKE TPU / TPU VM) | `DEVICE=tpu` | not on Modal/Azure; see `infra/gcp_tpu_vm.sh` |

### Modal (GPU + CPU)
```bash
pip install modal && modal deploy serverless/modal_app.py
modal run serverless/modal_app.py::main --task-id cholesky_update__gpu__f32 --device gpu --build
```

### Docker (any runtime)
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --build-arg DEVICE=cuda \
  --build-arg WITH_BUILD=1 -t jaxbench:cuda .
docker run --rm --gpus all \
  -v $PWD/bazel-disk:/cache/bazel-disk:ro -v $PWD/jax:/opt/jax \
  jaxbench:cuda cholesky_update__gpu__f32 --build
```

### TPU (GCP)
TPUs are Google-Cloud-only. Provision a TPU VM (`infra/gcp_tpu_vm.sh`) or deploy the
`DEVICE=tpu` image to Cloud Run TPU / GKE, then run the same `task_id`s whose `device`
is `tpu` (JAX runs the identical op on the TPU backend). The kernel *source* mutation
surface differs on TPU (XLA-codegen rather than these FFI kernels) — see the caveat in
`../docs/METHODOLOGY.md`.

## Recommendation

1. Build three device images (`cpu`/`cuda`/`tpu`) from `docker/Dockerfile`.
2. Seed a warm `bazel-disk` volume once (one cold build) and mount it read-only.
3. Drive evolution with ShinkaEvolve; fan candidates out to Modal (GPU+CPU) and a GCP
   TPU pool; collect `combined_score` centrally.
This keeps each candidate isolated and fast, and is the cheapest way to cover all
three device classes.
