#!/usr/bin/env bash
# Provision a GCP TPU VM as a JaxBench TPU device.
# TPUs are Google-Cloud-only (NOT available on Azure). PREREQ — authenticate first
# (interactive; an agent cannot do this for you):
#     gcloud auth login && gcloud config set project <PROJECT>
# Then:  bash infra/gcp_tpu_vm.sh
set -euo pipefail

ZONE=${ZONE:-us-central2-b}
NAME=${NAME:-jaxbench-tpu}
ACCEL=${ACCEL:-v5litepod-8}        # or v4-8, v3-8, v5p-8
VERSION=${VERSION:-tpu-ubuntu2204-base}

echo ">> project: $(gcloud config get-value project)"
gcloud compute tpus tpu-vm create "$NAME" \
  --zone="$ZONE" --accelerator-type="$ACCEL" --version="$VERSION"

cat <<EOF
>> TPU VM '$NAME' created ($ACCEL, $ZONE)
Run JaxBench on it:
  gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --command='
    sudo apt-get update && sudo apt-get install -y python3-pip git &&
    git clone https://github.com/Tyronita/JaxBench && cd JaxBench &&
    pip install -e ".[tpu]" &&
    JAX_ENABLE_X64=1 python -c "import jax; print(jax.devices())" &&
    JAX_ENABLE_X64=1 python shinka/make_baseline.py --device tpu &&
    JAX_ENABLE_X64=1 pytest tests/test_correctness.py -q -k tpu'
Tear down:  gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE --quiet
EOF
