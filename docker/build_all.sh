#!/usr/bin/env bash
# Build (and optionally push) all cross-device JaxBench images.
# REGISTRY is ANONYMISED — set JAXBENCH_REGISTRY to your own registry first.
#   az acr login -n <your-acr>            # auth first (interactive; you run this)
#   JAXBENCH_REGISTRY=<your-acr>.azurecr.io/jaxbench PUSH=1 bash docker/build_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${JAXBENCH_REGISTRY:=your-registry.azurecr.io/jaxbench}"   # anonymised default
TAG="${TAG:-0.2.0}"
DEVICES="${DEVICES:-cpu cuda tpu}"
PUSH="${PUSH:-0}"

for d in $DEVICES; do
  echo ">> $JAXBENCH_REGISTRY/runtime-$d:$TAG  (eval image)"
  DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --build-arg DEVICE="$d" \
    -t "$JAXBENCH_REGISTRY/runtime-$d:$TAG" -t "$JAXBENCH_REGISTRY/runtime-$d:latest" .
  echo ">> $JAXBENCH_REGISTRY/build-$d:$TAG   (build/toolchain image)"
  DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --build-arg DEVICE="$d" \
    --build-arg WITH_BUILD=1 -t "$JAXBENCH_REGISTRY/build-$d:$TAG" .
done

if [ "$PUSH" = "1" ]; then
  for d in $DEVICES; do
    docker push "$JAXBENCH_REGISTRY/runtime-$d:$TAG"; docker push "$JAXBENCH_REGISTRY/runtime-$d:latest"
    docker push "$JAXBENCH_REGISTRY/build-$d:$TAG"
  done
fi
echo "done. images for: $DEVICES (push=$PUSH)"
