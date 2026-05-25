#!/usr/bin/env python3
"""Record baseline latencies for every task on the current (unmutated) tree.

Writes shinka/baseline.json: {task_id: {N: seconds}}. ShinkaEvolve later scores a
mutated kernel as speedup = baseline_latency / candidate_latency. Run once before
evolving (and re-run if you change machines or the unmutated kernels).
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaxbench import registry, bench  # noqa: E402


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="only baseline tasks on this device")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "baseline.json"))
    a = ap.parse_args(argv)

    base = {}
    for t in registry.tasks():
        if a.device and t.device != a.device:
            continue
        try:
            ms = bench.sweep(t)
            base[t.id] = {str(m.n): m.latency_ms / 1e3 for m in ms}
            print(f"{t.id:36} " + "  ".join(f"N={m.n}:{m.latency_ms:.2f}ms" for m in ms))
        except Exception as e:  # e.g. GPU task with no GPU
            print(f"{t.id:36} skipped ({type(e).__name__})")
    with open(a.out, "w") as f:
        json.dump(base, f, indent=2)
    print(f"\nwrote {a.out} with {len(base)} task baselines")


if __name__ == "__main__":
    main()
