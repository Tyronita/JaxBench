# Provisioning the three device classes

JaxBench targets **GPU**, **TPU**, and **CPU**. These scripts stand up each device.
**Every cloud action needs interactive auth first** — an agent cannot run `az login` /
`gcloud auth login` for you; run them yourself, then the scripts.

| device | provider | script | hardware |
|---|---|---|---|
| GPU (A100/H100) | local box or any GPU cloud | — (run directly) | the A100 this repo was tuned on |
| CPU (many cores) | **Azure** | `az_cpu_vm.sh` | `Standard_F64s_v2` (64 vCPU) |
| TPU | **GCP only** | `gcp_tpu_vm.sh` | `v5litepod-8` (Azure has no TPUs) |

```bash
# CPU on Azure
az login                       # interactive — you run this
bash infra/az_cpu_vm.sh

# TPU on GCP (TPUs are not on Azure)
gcloud auth login              # interactive — you run this
gcloud config set project <PROJECT>
bash infra/gcp_tpu_vm.sh
```

For fully serverless execution (no long-lived VMs) use `../serverless/` instead:
Modal for GPU+CPU, GCP Cloud Run / GKE for TPU.

## Honest status of the 3-device runs

- **GPU (A100):** available on the tuning box; targeted-build + hot-swap recipe is
  validated. Real op latencies require the jaxlib/plugin wheels installed (see repo
  root README "GPU run").
- **CPU:** real results recorded on a 24-core box (`results/`), and reproducible on
  the Azure VM above.
- **TPU:** *not runnable from Azure.* Provision on GCP (`gcp_tpu_vm.sh`) to produce
  real TPU numbers. The device class, tasks, and commands are wired and ready; no TPU
  results are claimed until that run is done.
