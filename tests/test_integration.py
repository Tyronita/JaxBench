"""Larger, composite integration tests.

These exercise the kernels the way real workloads chain them, so a mutation that
passes the per-op gate but breaks a realistic pipeline is still caught.
"""
import os
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.scipy.linalg as jsl


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_lu_then_solve_matches_direct(n):
    """LU factor + lu_solve must match a direct solve across sizes."""
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal((n, 4))
    x_direct = jnp.linalg.solve(jnp.asarray(A), jnp.asarray(b))
    x_lu = jsl.lu_solve(jsl.lu_factor(jnp.asarray(A)), jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(x_direct), np.asarray(x_lu), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("n", [128, 512])
def test_cholesky_solve_spd(n):
    """SPD solve via Cholesky equals the generic solver."""
    rng = np.random.default_rng(n)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    b = rng.standard_normal((n, 1))
    L = jnp.linalg.cholesky(jnp.asarray(A))
    y = jsl.solve_triangular(L, jnp.asarray(b), lower=True)
    x = jsl.solve_triangular(L.conj().T, y, lower=False)
    np.testing.assert_allclose(np.asarray(jnp.asarray(A) @ x), b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", [64, 256])
def test_svd_low_rank_reconstruction(n):
    """Truncated SVD reconstructs a known low-rank matrix."""
    rng = np.random.default_rng(n)
    r = 8
    U = rng.standard_normal((n, r)); V = rng.standard_normal((r, n))
    A = U @ V
    u, s, vh = jnp.linalg.svd(jnp.asarray(A), full_matrices=False)
    A_r = (np.asarray(u)[:, :r] * np.asarray(s)[:r]) @ np.asarray(vh)[:r]
    np.testing.assert_allclose(A_r, A, rtol=1e-6, atol=1e-6)
    # singular values beyond the true rank are ~0
    assert np.max(np.asarray(s)[r:]) < 1e-6 * np.asarray(s)[0]


@pytest.mark.parametrize("n", [256, 4096])
def test_tridiagonal_solve_vs_dense(n):
    """Tridiagonal solver matches a dense solve of the same system."""
    la = jax.lax.linalg
    rng = np.random.default_rng(n)
    dl = rng.standard_normal(n); dl[0] = 0
    d = rng.standard_normal(n) + 4
    du = rng.standard_normal(n); du[-1] = 0
    b = rng.standard_normal((n, 1))
    x = la.tridiagonal_solve(jnp.asarray(dl), jnp.asarray(d), jnp.asarray(du), jnp.asarray(b))
    A = np.diag(d) + np.diag(du[:-1], 1) + np.diag(dl[1:], -1)
    np.testing.assert_allclose(np.asarray(A @ np.asarray(x)), b, rtol=1e-6, atol=1e-6)


def test_eigh_orthonormal_eigenvectors():
    """Eigenvectors of a Hermitian matrix are orthonormal and reconstruct it."""
    n = 256
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M @ M.conj().T
    w, V = jnp.linalg.eigh(jnp.asarray(A))
    V = np.asarray(V)
    np.testing.assert_allclose(V.conj().T @ V, np.eye(n), rtol=0, atol=1e-8)
    recon = (V * np.asarray(w)) @ V.conj().T
    np.testing.assert_allclose(recon, A, rtol=1e-6, atol=1e-6)
