import warnings

import numpy as np
import pytest

pyomo = pytest.importorskip("pyomo")

from opyrability import nlp_based_approach, AIS2AOS_map, implicit_map
from shower import shower2x2
from shower_pyomo_omlt import build_shower2x2


# --------------------------------------------------------------------------- #
# Pyomo/OMLT model support in the steady-state mapping functions (PR #33,
# adapted). The shower problem has both a callable (shower2x2) and an
# equation-oriented Pyomo builder (build_shower2x2) implementation, so the
# two paths must agree.
# --------------------------------------------------------------------------- #

DOS_BOUNDS = np.array([[10.0, 20.0], [70.0, 100.0]])
AIS_BOUNDS = np.array([[1.0, 10.0], [1.0, 10.0]])
U0 = np.array([5.0, 5.0])
LB = np.array([0.5, 0.5])
UB = np.array([20.0, 20.0])


class TestPyomoInverseMapping:

    def test_detection_and_agreement_with_callable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fDIS_p, fDOS_p, msg_p = nlp_based_approach(
                build_shower2x2, DOS_BOUNDS, [3, 3], U0, LB, UB, plot=False)
            fDIS_c, fDOS_c, _ = nlp_based_approach(
                shower2x2, DOS_BOUNDS, [3, 3], U0, LB, UB,
                method='ipopt', plot=False)
        np.testing.assert_allclose(fDOS_p, fDOS_c, atol=1e-3)
        np.testing.assert_allclose(fDIS_p, fDIS_c, atol=1e-3)
        assert all(m == 'optimal' for m in msg_p)

    def test_pyomo_solutions_satisfy_model(self):
        """The returned fDIS/fDOS rows must be consistent with the
        original callable model (the algebraic and callable forms are the
        same physics)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fDIS, fDOS, _ = nlp_based_approach(
                build_shower2x2, DOS_BOUNDS, [3, 3], U0, LB, UB, plot=False)
        for u_row, y_row in zip(fDIS, fDOS):
            np.testing.assert_allclose(shower2x2(u_row), y_row, atol=1e-5)

    def test_ad_and_constr_ignored_with_warning(self):
        with pytest.warns(UserWarning):
            nlp_based_approach(
                build_shower2x2, DOS_BOUNDS, [2, 2], U0, LB, UB,
                ad=True, constr={'type': 'ineq',
                                 'fun': lambda u: u[1] - u[0]},
                plot=False)


class TestPyomoForwardMapping:

    def test_proxy_agreement_with_callable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, AOS_p = AIS2AOS_map(build_shower2x2, AIS_BOUNDS, [3, 3],
                                   plot=False, output_dim=2)
            _, AOS_c = AIS2AOS_map(shower2x2, AIS_BOUNDS, [3, 3],
                                   plot=False)
        np.testing.assert_allclose(AOS_p, AOS_c, atol=1e-6)

    def test_output_dim_required(self):
        with pytest.raises(ValueError):
            AIS2AOS_map(build_shower2x2, AIS_BOUNDS, [3, 3], plot=False)


class TestPyomoImplicitMapGuard:

    def test_implicit_map_raises(self):
        with pytest.raises(NotImplementedError):
            implicit_map(build_shower2x2, image_init=np.array([2.0, 90.0]),
                         domain_bound=AIS_BOUNDS,
                         domain_resolution=[3, 3])
