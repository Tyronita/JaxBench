# 4-file serverless run — real fresh containers, real .so builds

Each result was produced by one fresh `docker run --rm` of the
`evolvebench.azurecr.io/jaxbench/build-cuda:0.2.0` image (the serverless pattern:
one isolated container per candidate), running the full eval loop on the A100:
stage candidate -> bazel build the cc_shared_library (`//jaxlib/cuda:_X.so`) ->
hot-swap into the venv plugin -> correctness vs stock -> latency sweep -> speedup
vs baseline -> restore. Candidates are no-op comment perturbations (no LLM key),
so scores cluster near 1.0 ± measurement noise — what's being demonstrated is
that the full loop works on real hardware, not a kernel optimisation win.
