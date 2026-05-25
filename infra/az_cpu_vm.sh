#!/usr/bin/env bash
# Provision a many-core Azure CPU VM as a JaxBench CPU device.
# PREREQ: you must authenticate first (interactive) — an agent cannot do this for you:
#     az login
# Then:  bash infra/az_cpu_vm.sh
set -euo pipefail

RG=${RG:-jaxbench-rg}
LOC=${LOC:-eastus}
VM=${VM:-jaxbench-cpu}
SIZE=${SIZE:-Standard_F64s_v2}     # 64 vCPU compute-optimised; F32s_v2=32, F72s_v2=72
IMAGE=${IMAGE:-Ubuntu2204}
ADMIN=${ADMIN:-azureuser}

echo ">> using subscription: $(az account show --query name -o tsv)"
az group create -n "$RG" -l "$LOC" -o none
az vm create -g "$RG" -n "$VM" --image "$IMAGE" --size "$SIZE" \
  --admin-username "$ADMIN" --generate-ssh-keys -o table
IP=$(az vm show -g "$RG" -n "$VM" -d --query publicIps -o tsv)
echo ">> VM ready at $IP ($SIZE)"
cat <<EOF
Next:
  ssh $ADMIN@$IP
  sudo apt-get update && sudo apt-get install -y python3-pip git
  git clone https://github.com/Tyronita/JaxBench && cd JaxBench
  pip install -e ".[cpu]"
  JAX_ENABLE_X64=1 python shinka/make_baseline.py --device cpu
  JAX_ENABLE_X64=1 pytest tests/ -q
Tear down:  az group delete -n $RG --yes --no-wait
EOF
