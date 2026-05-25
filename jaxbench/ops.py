"""Operation layer: build well-conditioned inputs and run the JAX op under test.

`make_inputs(op, n, dtype, seed)` returns host (numpy) arrays so the same problem
feeds both the JAX op (device) and the reference (host). `jax_callable(op)` returns
a jitted function executing the op exactly as a user would call it, so it dispatches
through the jaxlib kernel that ShinkaEvolve mutates.
"""
from __future__ import annotations
import numpy as np

NP_DTYPE = {"f32": np.float32, "f64": np.float64,
            "c64": np.complex64, "c128": np.complex128, "i32": np.int32}
REAL_OF = {"f32": np.float32, "f64": np.float64,
           "c64": np.float32, "c128": np.float64}


def _rng(seed): return np.random.default_rng(seed)


def _randmat(rng, n, dtype):
    d = NP_DTYPE[dtype]
    if dtype in ("c64", "c128"):
        r = REAL_OF[dtype]
        return (rng.standard_normal((n, n)).astype(r)
                + 1j * rng.standard_normal((n, n)).astype(r)).astype(d)
    return rng.standard_normal((n, n)).astype(d)


def _spd(rng, n, dtype):
    """Symmetric/Hermitian positive-definite matrix."""
    a = _randmat(rng, n, dtype)
    h = a @ a.conj().T + n * np.eye(n, dtype=NP_DTYPE[dtype])
    return h.astype(NP_DTYPE[dtype])


def make_inputs(op: str, n: int, dtype: str, seed: int = 0) -> dict:
    rng = _rng(seed)
    if op in ("cholesky",):
        return {"a": _spd(rng, n, dtype)}
    if op in ("eigh",):
        return {"a": _spd(rng, n, dtype)}
    if op in ("lu", "qr", "svd"):
        return {"a": _randmat(rng, n, dtype)}
    if op == "inv":
        return {"a": _randmat(rng, n, dtype) + n * np.eye(n, dtype=NP_DTYPE[dtype])}
    if op in ("det", "slogdet"):
        # near-identity so |det| stays O(1) (a random matrix's det overflows f32)
        return {"a": np.eye(n, dtype=NP_DTYPE[dtype]) + (0.3 / np.sqrt(n)) * _randmat(rng, n, dtype)}
    if op == "expm":
        # modest-norm matrix so e^A does not overflow
        return {"a": (1.0 / n) * _randmat(rng, n, dtype)}
    if op in ("solve", "lu_solve"):
        a = _randmat(rng, n, dtype) + n * np.eye(n, dtype=NP_DTYPE[dtype])
        b = _randmat(rng, n, dtype)[:, :1]
        return {"a": a, "b": b}
    if op == "lstsq":
        a = _randmat(rng, n, dtype)
        b = _randmat(rng, n, dtype)[:, :1]
        return {"a": a, "b": b}
    if op == "cholesky_update":
        return {"r": np.linalg.cholesky(_spd(rng, n, dtype)).conj().T.astype(NP_DTYPE[dtype]),
                "w": _randmat(rng, n, dtype)[:, 0]}
    if op == "lu_pivots_to_permutation":
        return {"pivots": rng.integers(0, n, size=(n,), dtype=np.int32), "n": n}
    if op == "householder_product":
        a = _randmat(rng, n, dtype)
        taus = rng.standard_normal((n,)).astype(REAL_OF[dtype]).astype(NP_DTYPE[dtype])
        return {"a": a, "taus": taus}
    if op == "qr_pivoting":
        return {"a": _randmat(rng, n, dtype)}
    if op in ("tridiagonal_solve", "tridiagonal_solve_perturbed"):
        dl = rng.standard_normal(n).astype(NP_DTYPE[dtype]); dl[0] = 0
        d = (rng.standard_normal(n) + 4).astype(NP_DTYPE[dtype])  # diagonally dominant
        du = rng.standard_normal(n).astype(NP_DTYPE[dtype]); du[-1] = 0
        b = rng.standard_normal((n, 1)).astype(NP_DTYPE[dtype])
        return {"dl": dl, "d": d, "du": du, "b": b}
    if op in ("threefry_uniform", "threefry_normal"):
        return {"shape": (n,), "seed": seed}
    if op == "csr_matmul":
        dens = 0.02
        a = rng.standard_normal((n, n)).astype(NP_DTYPE[dtype])
        a[rng.random((n, n)) > dens] = 0
        x = rng.standard_normal((n, 16)).astype(NP_DTYPE[dtype])
        return {"a": a, "x": x}
    raise KeyError(f"no input generator for op {op!r}")


def jax_callable(op: str):
    """Return a jitted function f(**device_inputs) -> output dispatching to jaxlib."""
    import jax, jax.numpy as jnp
    import jax.scipy.linalg as jsl
    la = jax.lax.linalg

    def jit(f): return jax.jit(f)

    table = {
        "lu":       jit(lambda a: jsl.lu(a)),
        "qr":       jit(lambda a: jnp.linalg.qr(a)),
        "svd":      jit(lambda a: jnp.linalg.svd(a, full_matrices=False)),
        "eigh":     jit(lambda a: jnp.linalg.eigh(a)),
        "solve":    jit(lambda a, b: jnp.linalg.solve(a, b)),
        "inv":      jit(lambda a: jnp.linalg.inv(a)),
        "cholesky": jit(lambda a: jnp.linalg.cholesky(a)),
        "det":      jit(lambda a: jnp.linalg.det(a)),
        "slogdet":  jit(lambda a: jnp.linalg.slogdet(a)),
        "expm":     jit(lambda a: jsl.expm(a)),
        "lstsq":    jit(lambda a, b: jnp.linalg.lstsq(a, b)[0]),
        "lu_solve": jit(lambda a, b: jsl.lu_solve(jsl.lu_factor(a), b)),
        "cholesky_update": jit(lambda r, w: la.cholesky_update(r, w)),
        "lu_pivots_to_permutation":
            (lambda pivots, n: la.lu_pivots_to_permutation(jnp.asarray(pivots), n)),
        "householder_product": jit(lambda a, taus: la.householder_product(a, taus)),
        "qr_pivoting": jit(lambda a: jsl.qr(a, pivoting=True)),
        "tridiagonal_solve": jit(lambda dl, d, du, b: la.tridiagonal_solve(dl, d, du, b)),
        "tridiagonal_solve_perturbed": jit(lambda dl, d, du, b: la.tridiagonal_solve(dl, d, du, b)),
        "csr_matmul": jit(lambda a, x: a @ x),
    }
    if op in ("threefry_uniform", "threefry_normal"):
        def _prng(shape, seed):
            k = jax.random.PRNGKey(seed)
            fn = jax.random.uniform if op == "threefry_uniform" else jax.random.normal
            return jax.jit(lambda key: fn(key, shape))(k)
        return _prng
    return table[op]
