import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from opyrability import implicit_map
from shower import shower2x2

# --------------------------------------------------------------------------- #
# implicit_map (predictor-corrector continuation of F(u, y) = 0).
#
# The suite previously only checked that implicit_map RAISES on a Pyomo model
# (test_pyomo_mapping.py); its actual continuation algorithm was never run, so
# correctness was unverified -- and it had a seeding bug (the continuation was
# seeded at domain_bound[0, :], the [lo, hi] of input 0, but stored in grid
# cell (0, 0), whose input is the lower-bound corner domain_bound[:, 0]). The
# seed is now placed at the correct grid cell, and these tests pin the real
# contract: every solved point must satisfy F(u, y) = 0 to the corrector
# tolerance, which for a forward map means y equals the model output.
# --------------------------------------------------------------------------- #


def _solve(F, image_init, domain_bound, resolution):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = implicit_map(F, image_init=np.asarray(image_init, float),
                           domain_bound=domain_bound,
                           domain_resolution=resolution,
                           direction='forward')
    return np.asarray(out[0]), np.asarray(out[1])


def test_implicit_map_linear_is_exact():
    """For a linear map y = A u the continuation is exact at every grid cell."""
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    def F(u, y):  # F(u, y) = 0  <=>  y = A u
        return jnp.array([y[0] - (A[0, 0] * u[0] + A[0, 1] * u[1]),
                          y[1] - (A[1, 0] * u[0] + A[1, 1] * u[1])])

    domain = np.array([[0.0, 1.0], [0.0, 1.0]])
    # Seed input is the lower-bound corner domain_bound[:, 0]; its output is A u.
    domain_set, image_set = _solve(F, A @ domain[:, 0], domain, [4, 4])

    n_solved = 0
    for idx in np.ndindex(domain_set.shape[:-1]):
        u, y = domain_set[idx], image_set[idx]
        if np.isnan(y).any() or np.isnan(u).any():
            continue
        n_solved += 1
        # On F = 0, and matching the explicit forward map.
        assert np.abs(np.asarray(F(jnp.array(u), jnp.array(y)))).max() < 1e-9
        np.testing.assert_allclose(y, A @ u, atol=1e-9)
    assert n_solved == 16  # 4 x 4 grid, all cells solved


def test_implicit_map_nonlinear_shower_on_solution_manifold():
    """For the (curved) shower map every solved point lies on F(u, y) = 0 and
    reproduces the explicit model output."""
    def F(u, y):
        return jnp.array([y[0] - (u[0] + u[1]),
                          y[1] * (u[0] + u[1]) - (60.0 * u[0] + 120.0 * u[1])])

    domain = np.array([[1.0, 10.0], [1.0, 10.0]])
    domain_set, image_set = _solve(F, shower2x2(domain[:, 0]), domain, [5, 5])

    n_solved = 0
    for idx in np.ndindex(domain_set.shape[:-1]):
        u, y = domain_set[idx], image_set[idx]
        if np.isnan(y).any() or np.isnan(u).any():
            continue
        n_solved += 1
        assert np.abs(np.asarray(F(jnp.array(u), jnp.array(y)))).max() < 1e-6
        np.testing.assert_allclose(y, shower2x2(u), atol=1e-6)
    assert n_solved == 25
