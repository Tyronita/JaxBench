# Cross-device image registry (anonymised)

Every device class and setup has a prebuilt image so the serverless make-change →
rebuild → verify loop is ready everywhere. **Registry refs are anonymised** — set
`JAXBENCH_REGISTRY` to your own (e.g. an Azure Container Registry) before building.

```
REGISTRY = ${JAXBENCH_REGISTRY:-your-registry.azurecr.io/jaxbench}    # anonymised
```

| device | role | image | base | used for |
|---|---|---|---|---|
| **GPU** (A100/H100) | eval  | `${REGISTRY}/runtime-cuda:0.2.0` | `nvidia/cuda:12.9` + `jax[cuda12]` | run challenge speed-up tests on GPU |
| **GPU** | build | `${REGISTRY}/build-cuda:0.2.0`  | + bazel + clang | rebuild a mutated kernel `.so` / XLA wheel |
| **CPU** (many-core) | eval  | `${REGISTRY}/runtime-cpu:0.2.0`  | `python:3.12-slim` + `jax[cpu]` | run challenge tests on CPU |
| **CPU** | build | `${REGISTRY}/build-cpu:0.2.0`   | + bazel + clang | rebuild CPU `_lapack.so` / XLA wheel |
| **TPU** | eval  | `${REGISTRY}/runtime-tpu:0.2.0`  | `python:3.12` + `jax[tpu]` (GCP) | run challenge tests on TPU |
| **TPU** | build | `${REGISTRY}/build-tpu:0.2.0`   | + bazel | rebuild for TPU (GCP) |

Build them all (one command):
```bash
JAXBENCH_REGISTRY=<your-acr>.azurecr.io/jaxbench docker buildx bake -f docker/bake.hcl all
# or: bash docker/build_all.sh         # plain docker build, all devices
# push: ... --push     |     PUSH=1 bash docker/build_all.sh
```

Run a challenge in a container (warm cache mounted read-only, jax checkout mounted):
```bash
docker run --rm --gpus all \
  -v $PWD/bazel-disk:/cache/bazel-disk:ro -v $PWD/jax:/opt/jax \
  ${REGISTRY}/build-cuda:0.2.0 \
  python -m jaxbench.challenge_runner prng_threefry --device gpu --build
```

Notes:
- Eval images are light; **build** images carry bazel+clang and are what the
  evolutionary workers use to rebuild a candidate.
- TPU images are for **GCP** (Cloud Run TPU / GKE / TPU VM) — Azure has no TPUs.
- The real registry name is intentionally not committed; override via the env var.
