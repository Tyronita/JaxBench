# Driving JaxBench with ShinkaEvolve

ShinkaEvolve is the outer loop; JaxBench is the evaluator. Each island is one jaxlib
mutation-surface file. The loop per candidate:

```
ShinkaEvolve mutates code inside // EVOLVE-BLOCK-START/END
        │
        ▼
shinka/evaluate.py --file <that file>
        │  build the one affected extension .so   (clang device compiler, ~5–13s)
        │  hot-swap the .so into the venv          (no wheel repackage)
        │  correctness gate  ── fail ─▶ fitness = 0.0
        │  latency sweep → speedup vs baseline
        ▼
fitness (mean speedup) on stdout  ── ShinkaEvolve maximises this
```

## 1. Record the baseline (once, on the unmutated tree)

```bash
python shinka/make_baseline.py            # writes shinka/baseline.json
```

## 2. Point ShinkaEvolve at the islands

`shinka_config.yaml` defines one island per file, ordered by rebuild cost so the
cheapest-to-iterate, highest-value kernels (`prng`, `linalg`, `solver`) evolve first.
Each island's `eval_cmd` is the JaxBench evaluator.

## 3. Fitness contract

- **Correctness is a hard gate**: if *any* task for the file fails its residual
  tolerance, fitness is `0.0` (a fast-but-wrong kernel cannot win).
- Otherwise fitness = **mean speedup** across that file's tasks, over the full
  (N × dtype) sweep, vs the recorded baseline.

## 4. Why this is fast

The evaluator never rebuilds the wheel. It rebuilds only the affected extension
`.so` with clang as the CUDA device compiler and hot-swaps it — the recipe measured
at ~13s for a device kernel vs ~21s for a full nvcc wheel build. See
[`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).

## 5. Mutation hygiene

ShinkaEvolve only edits between `// EVOLVE-BLOCK-START` and `// EVOLVE-BLOCK-END`.
Keep the surface in small leaf `.cc`/`.cu.cc` files; avoid shared headers (they fan
out across extensions and wheels). See the build/cost notes in the methodology doc.
