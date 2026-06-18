import warnings

import numpy as np
import pytest

pounce = pytest.importorskip("pounce")

from opyrability import nlp_based_approach
from shower import shower2x2


# --------------------------------------------------------------------------- #
# Pounce solver support (method='pounce') in the inverse mapping. Pounce is
# the pure-Rust reimplementation of the Ipopt interior-point solver with the
# bundled FERAL linear solver (https://kitchingroup.cheme.cmu.edu/pounce/).
# The inverse problems must agree with the cyipopt reference.
# --------------------------------------------------------------------------- #

DOS_BOUNDS = np.array([[10.0, 20.0], [70.0, 100.0]])
U0 = np.array([5.0, 5.0])
LB = np.array([0.5, 0.5])
UB = np.array([20.0, 20.0])


def _solve(method, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nlp_based_approach(shower2x2, DOS_BOUNDS, [3, 3], U0, LB, UB,
                                  method=method, plot=False, **kwargs)


class TestPounceInverseMapping:

    def test_numerical_derivatives_agree_with_ipopt(self):
        """ad=False: Pounce uses internal central finite differences."""
        fDIS_p, fDOS_p, msg_p = _solve('pounce')
        fDIS_i, fDOS_i, _ = _solve('ipopt')
        np.testing.assert_allclose(fDIS_p, fDIS_i, atol=1e-3)
        np.testing.assert_allclose(fDOS_p, fDOS_i, atol=1e-3)
        assert all('Succeeded' in m or 'success' in m.lower()
                   for m in msg_p)

    def test_ad_jacobian_and_hessian_agree_with_ipopt(self):
        """ad=True, bound-constrained: Pounce consumes the JAX gradient
        AND the JAX Hessian (which cyipopt currently cannot)."""
        import jax.numpy as jnp

        def shower_jax(u):
            return jnp.array([u[0] + u[1],
                              (u[0] * 60 + u[1] * 120) / (u[0] + u[1])])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fDIS_p, fDOS_p, msg_p = nlp_based_approach(
                shower_jax, DOS_BOUNDS, [3, 3], U0, LB, UB,
                method='pounce', ad=True, plot=False)
        _, fDOS_i, _ = _solve('ipopt')
        np.testing.assert_allclose(np.asarray(fDOS_p), fDOS_i, atol=1e-3)
        assert all('Succeeded' in m for m in msg_p)

    def test_constraint_dict_honored(self):
        """Constrained problems use the same constraint dict format as
        cyipopt."""
        con = {'type': 'ineq', 'fun': lambda u: u[1] - u[0]}
        fDIS, _, msg = _solve('pounce', constr=con)
        assert np.all(fDIS[:, 1] - fDIS[:, 0] >= -1e-6)
        assert all('Succeeded' in m for m in msg)

    def test_bounds_respected(self):
        fDIS, _, _ = _solve('pounce')
        assert np.all(fDIS >= LB - 1e-8)
        assert np.all(fDIS <= UB + 1e-8)
