// docker buildx bake — build all cross-device JaxBench images in one shot.
//   REGISTRY is ANONYMISED; override with your own (e.g. an Azure Container Registry):
//     REGISTRY=myacr.azurecr.io/jaxbench docker buildx bake -f docker/bake.hcl
//   build everything (eval + build/toolchain variants, all devices):
//     docker buildx bake -f docker/bake.hcl all
//   push to your registry:
//     docker buildx bake -f docker/bake.hcl all --push

variable "REGISTRY" { default = "your-registry.azurecr.io/jaxbench" }  // anonymised placeholder
variable "TAG"      { default = "0.2.0" }

group "default" { targets = ["cpu", "cuda", "tpu"] }                 // eval images
group "all"     { targets = ["cpu","cuda","tpu","cpu-build","cuda-build","tpu-build"] }

target "_common" {
  context    = ".."
  dockerfile = "docker/Dockerfile"
}

// eval images (run correctness + perf on each device)
target "cpu"  { inherits=["_common"] args={DEVICE="cpu"}  tags=["${REGISTRY}/runtime-cpu:${TAG}",  "${REGISTRY}/runtime-cpu:latest"]  }
target "cuda" { inherits=["_common"] args={DEVICE="cuda"} tags=["${REGISTRY}/runtime-cuda:${TAG}", "${REGISTRY}/runtime-cuda:latest"] }
target "tpu"  { inherits=["_common"] args={DEVICE="tpu"}  tags=["${REGISTRY}/runtime-tpu:${TAG}",  "${REGISTRY}/runtime-tpu:latest"]  }

// build images (add bazel + clang so they can rebuild a mutated kernel/.so)
target "cpu-build"  { inherits=["_common"] args={DEVICE="cpu",  WITH_BUILD="1"} tags=["${REGISTRY}/build-cpu:${TAG}"]  }
target "cuda-build" { inherits=["_common"] args={DEVICE="cuda", WITH_BUILD="1"} tags=["${REGISTRY}/build-cuda:${TAG}"] }
target "tpu-build"  { inherits=["_common"] args={DEVICE="tpu",  WITH_BUILD="1"} tags=["${REGISTRY}/build-tpu:${TAG}"]  }
