"""Correctness references.

Decompositions are not unique (QR column signs, SVD/eigh sign & order), so we check
the defining *mathematical property* (a residual) rather than raw factor equality.
`residual(op, inputs, out)` returns a scalar relative error; the task passes when it
is below the dtype tolerance. Inputs and `out` are host numpy arrays.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = max(np.linalg.norm(b.ravel()), 1.0)
    return float(np.linalg.norm((a - b).ravel()) / denom)


def residual(op: str, inp: dict, out) -> float:
    if op == "lu":                       # out = (P, L, U) from scipy-style lu
        P, L, U = out
        return _rel(np.asarray(P) @ np.asarray(L) @ np.asarray(U), inp["a"])
    if op == "qr":
        Q, R = out
        return _rel(np.asarray(Q) @ np.asarray(R), inp["a"])
    if op == "svd":
        U, s, Vh = out
        A = (np.asarray(U) * np.asarray(s)) @ np.asarray(Vh)
        return _rel(A, inp["a"])
    if op == "eigh":
        w, V = out
        V = np.asarray(V); w = np.asarray(w)
        return _rel(inp["a"] @ V, V @ np.diag(w))
    if op == "cholesky":
        L = np.asarray(out)
        return _rel(L @ L.conj().T, inp["a"])
    if op in ("solve", "lu_solve"):
        return _rel(inp["a"] @ np.asarray(out), inp["b"])
    if op == "inv":
        return _rel(inp["a"] @ np.asarray(out), np.eye(inp["a"].shape[0], dtype=inp["a"].dtype))
    if op == "det":
        return _rel(np.asarray(out), np.linalg.det(inp["a"].astype(np.complex128)))
    if op == "slogdet":
        sign, logabs = out
        ref = np.linalg.slogdet(inp["a"].astype(np.complex128))
        return max(_rel(sign, ref[0]), _rel(logabs, ref[1]))
    if op == "expm":
        return _rel(np.asarray(out), sla.expm(inp["a"]))
    if op == "lstsq":
        ref = np.linalg.lstsq(inp["a"], inp["b"], rcond=None)[0]
        return _rel(np.asarray(out), ref)
    if op == "cholesky_update":
        R = inp["r"]; w = inp["w"]
        A = R.conj().T @ R + np.outer(w, w.conj())
        Rn = np.asarray(out)
        return _rel(Rn.conj().T @ Rn, A)
    if op == "lu_pivots_to_permutation":
        # property: result is a permutation of range(n)
        perm = np.sort(np.asarray(out).ravel())
        return _rel(perm, np.arange(len(perm)))
    if op == "householder_product":
        # property: result has orthonormal columns (Q^H Q = I)
        Q = np.asarray(out)
        return _rel(Q.conj().T @ Q, np.eye(Q.shape[1], dtype=Q.dtype))
    if op == "qr_pivoting":
        Q, R, piv = out
        A = inp["a"][:, np.asarray(piv)]
        return _rel(np.asarray(Q) @ np.asarray(R), A)
    if op in ("tridiagonal_solve", "tridiagonal_solve_perturbed"):
        dl, d, du, b = inp["dl"], inp["d"], inp["du"], inp["b"]
        n = len(d)
        A = np.diag(d) + np.diag(du[:-1], 1) + np.diag(dl[1:], -1)
        return _rel(A @ np.asarray(out), b)
    if op == "csr_matmul":
        return _rel(np.asarray(out), inp["a"] @ inp["x"])
    if op in ("threefry_uniform", "threefry_normal"):
        x = np.asarray(out).ravel()
        if op == "threefry_uniform":
            return float(max(abs(x.mean() - 0.5), abs(x.var() - 1/12)) * 3)  # statistical
        return float(max(abs(x.mean()), abs(x.var() - 1.0)) * 0.5)
    raise KeyError(f"no reference for op {op!r}")
