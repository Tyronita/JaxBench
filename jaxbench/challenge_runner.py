"""Evaluate a JaxBench challenge: build the candidate, run its speed-up test, gate on
correctness vs stock JAX, score the speedup. Tier- and device-aware.

  jaxlib_kernel:  build the extension .so (clang device) -> hot-swap -> test
  xla_core:       build the plugin wheel against the editable XLA copy
                  (--override_repository); applying needs a reinstall, which the
                  serverless model does by shipping a fresh container per candidate.

The speed-up test is a JAX workload (jaxbench.workloads) that routes through the hot
path; correctness compares the rebuilt result to a trusted host reference.
"""
from __future__ import annotations
import os, shutil, statistics, time
from dataclasses import asdict

from . import challenges as _ch
from . import workloads as _wl
from . import build as _build


def _measure(workload: str, size: int, dtype: str, device: str,
             warmup: int = 3, trials: int = 15) -> float:
    import jax, jax.numpy as jnp
    fn = _wl.fn(workload)
    host = _wl.inputs(workload, size, dtype, 0)
    dev = jax.devices(device)[0]
    dinp = {}
    for k, v in host.items():
        try:
            dinp[k] = jax.device_put(jnp.asarray(v), dev) if hasattr(v, "shape") else v
        except Exception:
            dinp[k] = v
    call = lambda: jax.block_until_ready(fn(**dinp))
    for _ in range(warmup):
        call()
    s = []
    for _ in range(trials):
        t0 = time.perf_counter(); call(); s.append(time.perf_counter() - t0)
    return statistics.median(s)


def _correct(workload: str, sizes, dtype: str, tol: float = 1e-3) -> tuple[bool, float]:
    import jax, numpy as np
    fn = _wl.fn(workload)
    worst = 0.0
    for n in sizes:
        host = _wl.inputs(workload, n, dtype, 1)
        out = jax.block_until_ready(fn(**host))
        if isinstance(out, tuple):
            out = tuple(np.asarray(o) for o in out)
        else:
            out = np.asarray(out)
        worst = max(worst, _wl.reference(workload, host, out))
    return worst <= tol, worst


def evaluate(challenge, *, device: str = "gpu", do_build: bool = False,
             candidate_source: str | None = None, baseline: dict | None = None,
             seed: int = 0) -> dict:
    c = challenge if hasattr(challenge, "id") else _ch.by_id(challenge)
    if device not in c.test.devices:
        return {"challenge": c.id, "skipped": f"device {device} not in {c.test.devices}"}
    rec = {"challenge": c.id, "tier": c.tier, "file": c.file, "device": device,
           "build_target": c.build.target}

    backup = None
    try:
        if do_build:
            if c.tier == "xla_core":
                if candidate_source:
                    backup = ("xla", _build.stage_xla_candidate(c.file, candidate_source))
                br = _build.build_target(c.build.target, xla_override=True,
                                         log_path=None)
            else:
                dest = os.path.join(_build.JAX_REPO, c.file)
                if candidate_source:
                    backup = ("repo", dest + ".bak")
                    shutil.copy2(dest, backup[1]); open(dest, "w").write(candidate_source)
                br = _build.build_target(c.build.target,
                                         clang_device=c.file.endswith(".cu.cc"))
                if br.rc == 0 and c.build.so_path:
                    _build.hot_swap(br.so_src, c.build.so_path)
            rec["build"] = {"rc": br.rc, "wall_s": round(br.wall_s, 2),
                            "elapsed_s": br.elapsed_s, "command": c.build.command()}
            if br.rc != 0:
                rec["combined_score"] = 0.0; rec["error"] = "build failed"; return rec

        dt = c.test.dtypes[0]
        ok, worst = _correct(c.test.workload, c.test.sizes, dt)
        rec["correctness"] = {"passed": ok, "worst_residual": worst}
        if not ok:
            rec["combined_score"] = 0.0; return rec

        perf, speedups = [], []
        for n in c.test.sizes:
            lat = _measure(c.test.workload, n, dt, device)
            perf.append({"size": n, "latency_ms": round(lat * 1e3, 4)})
            if baseline and str(n) in baseline:
                speedups.append(baseline[str(n)] / lat)
        rec["perf"] = perf
        rec["combined_score"] = round(sum(speedups) / len(speedups), 4) if speedups else 1.0
        return rec
    finally:
        if backup:
            kind, path = backup
            if kind == "repo":
                dest = os.path.join(_build.JAX_REPO, c.file)
                shutil.copy2(path, dest); os.remove(path)
                if c.build.so_path:
                    try: _build.restore(c.build.so_path)
                    except Exception: pass
            else:
                _build.restore_xla(c.file)


def main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser(description="Evaluate a JaxBench challenge.")
    ap.add_argument("challenge_id")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(evaluate(_ch.by_id(a.challenge_id), device=a.device, do_build=a.build),
                     indent=2))


if __name__ == "__main__":
    main()
