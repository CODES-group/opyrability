import warnings

import numpy as np
import pytest

from opyrability import (nlp_based_approach, milp_based_approach,
                         AIS2AOS_map, points2simplices, _paired_simplices)
from shower import shower2x2


# --------------------------------------------------------------------------- #
# P2/P3 process intensification problems (Carrasco and Lima) inside
# nlp_based_approach, and the multilayer MILP (Gazzaneo and Lima). The
# shower problem gives analytic optima: y1 = u1 + u2 exactly, so footprint
# style targets have closed-form solutions.
# --------------------------------------------------------------------------- #

U0 = np.array([5.0, 5.0])
LB = np.array([0.0, 0.0])
UB = np.array([100.0, 100.0])
DOS = np.array([[5.0, 10.0], [80.0, 100.0]])


def _solve(problem, PI_target, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nlp_based_approach(shower2x2, DOS, [6, 6], U0, LB, UB,
                                  method='ipopt', plot=False,
                                  problem=problem, PI_target=PI_target,
                                  **kwargs)


class TestP2:

    def test_total_flow_target_analytic_optimum(self):
        """PI = u1 + u2 = y1 exactly; with y1 >= 5 in the DOS the optimal
        intensified design has PI = 5."""
        fDIS, fDOS, msgs, rep = _solve('P2', lambda u: u[0] + u[1])
        assert rep['PI_value'] == pytest.approx(5.0, abs=1e-4)
        assert rep['problem'] == 'P2'
        # The intensified outputs lie inside the performance box.
        assert np.all(rep['y_PI'] >= DOS[:, 0] - 1e-6)
        assert np.all(rep['y_PI'] <= DOS[:, 1] + 1e-6)
        # And are consistent with the model.
        np.testing.assert_allclose(shower2x2(rep['u_PI']), rep['y_PI'],
                                   atol=1e-6)

    def test_pi_bounds_filters_performance_subset(self):
        """A tighter level of performance (y2 >= 90) shrinks DOS_PI and
        the optimum respects it."""
        PI_bounds = np.array([[5.0, 10.0], [90.0, 100.0]])
        _, _, _, rep = _solve('P2', lambda u: u[0] + u[1],
                              PI_bounds=PI_bounds)
        assert np.all(rep['DOS_PI'][:, 1] >= 90.0 - 1e-6)
        assert rep['y_PI'][1] >= 90.0 - 1e-6
        # DIS_PI/DOS_PI are subsets of the full DIS*/DOS* grid results.
        assert rep['DIS_PI'].shape[0] <= 36

    def test_returns_three_tuple_for_p1(self):
        """Backwards compatibility: the default problem returns exactly
        the classic 3-tuple."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = nlp_based_approach(shower2x2, DOS, [3, 3], U0, LB, UB,
                                     method='ipopt', plot=False)
        assert len(out) == 3

    def test_pi_target_required(self):
        with pytest.raises(ValueError):
            _solve('P2', None)

    def test_invalid_problem_raises(self):
        with pytest.raises(ValueError):
            _solve('bogus', lambda u: u[0])

    def test_empty_performance_subset_raises(self):
        unreachable = np.array([[500.0, 600.0], [500.0, 600.0]])
        with pytest.raises(ValueError):
            _solve('P2', lambda u: u[0] + u[1], PI_bounds=unreachable)


class TestP3:

    def test_bilevel_equals_sequential(self):
        """The dissertation's validation (p. 60): P3 (the bilevel
        program, solved by its sequential equivalence) gives the same
        result as P2, since both solve the inner P1 over the full DOS
        grid and then minimize the PI target over the feasible region."""
        PI = lambda u: u[0] + u[1]
        _, _, _, rep3 = _solve('P3', PI)
        _, _, _, rep2 = _solve('P2', PI)
        assert rep3['PI_value'] == pytest.approx(rep2['PI_value'],
                                                 abs=1e-12)
        np.testing.assert_allclose(rep3['u_PI'], rep2['u_PI'], atol=1e-12)

    def test_outer_never_worse_than_grid(self):
        _, _, _, rep = _solve('P3', lambda u: u[0])
        assert rep['PI_value'] <= rep['PI_grid'].min() + 1e-12


# --------------------------------------------------------------------------- #
# Multilayer MILP (Gazzaneo and Lima, IECR 2019, Layer 1). With the linear
# PI target u1 + u2 = y1, the barycentric linearization is exact and the
# optimal phi equals the active DOS bound on y1.
# --------------------------------------------------------------------------- #

MILP_AIS = np.array([[0.1, 10.0], [0.1, 10.0]])
MILP_DOS = np.array([[6.0, 9.0], [85.0, 95.0]])
MILP_CONSTR = (np.array([[1.0, -1.0]]), np.array([0.0]))  # u1 <= u2


def _solve_milp(**kwargs):
    defaults = dict(AIS_bound=MILP_AIS,
                    PI_target=lambda u: u[0] + u[1],
                    DOS_bounds=MILP_DOS,
                    AIS_resolution=3,
                    input_constr=MILP_CONSTR,
                    plot=False)
    defaults.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return milp_based_approach(shower2x2, **defaults)


class TestMILP:

    def test_linear_target_exact_optimum(self):
        """y1 = u1 + u2 exactly, so phi must converge to the active DOS
        bound y1 >= 6."""
        u_opt, y_opt, phi, pi_true, hist = _solve_milp()
        assert phi == pytest.approx(6.0, abs=1e-6)
        # For a linear PI target the linearized and true values coincide.
        assert pi_true == pytest.approx(phi, abs=1e-6)
        # phi is a valid lower-bounded objective at every iteration.
        assert all(h['phi'] >= 6.0 - 1e-9 for h in hist)
        # Exact for the linear output: model(u_opt)[0] == y_opt[0].
        assert shower2x2(u_opt)[0] == pytest.approx(y_opt[0], abs=1e-9)
        assert MILP_DOS[0, 0] - 1e-9 <= y_opt[0] <= MILP_DOS[0, 1] + 1e-9

    def test_input_constraint_honored(self):
        u_opt, _, _, _, _ = _solve_milp()
        A_u, b_u = MILP_CONSTR
        assert np.all(A_u @ u_opt <= b_u + 1e-9)

    def test_history_contract_and_bound_refinement(self):
        _, _, _, _, hist = _solve_milp(tol=1e-9, max_iter=4)
        keys = {'phi', 'u', 'y', 'PI_true', 'AIS_simplex', 'AOS_simplex',
                'AIS_bound', 'n_pairs', 'E_rel', 'milp_status'}
        for h in hist:
            assert keys <= set(h.keys())
        # Bounds of iteration i+1 equal the winning simplex bounding box
        # of iteration i (up to the degeneracy width guard).
        for h_prev, h_next in zip(hist[:-1], hist[1:]):
            V = h_prev['AIS_simplex']
            np.testing.assert_allclose(h_next['AIS_bound'][:, 0],
                                       V.min(axis=1), atol=1e-9)
            assert np.all(h_next['AIS_bound'][:, 1]
                          >= V.max(axis=1) - 1e-9)

    def test_infeasible_dos_raises(self):
        with pytest.raises(ValueError):
            _solve_milp(DOS_bounds=np.array([[500.0, 510.0],
                                             [500.0, 510.0]]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            _solve_milp(DOS_bounds=np.array([[6.0, 9.0]]))


class TestPairedSimplices:

    def test_public_points2simplices_unchanged(self):
        """Golden regression: the refactor through _paired_simplices must
        keep the public output byte-identical."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AIS, AOS = AIS2AOS_map(shower2x2,
                                   np.array([[1.0, 10.0], [1.0, 10.0]]),
                                   [3, 3], plot=False)
        ais_s, aos_s = points2simplices(AIS, AOS)
        assert len(ais_s) == 8
        np.testing.assert_allclose(
            ais_s[0], np.array([[1.0, 1.0], [1.0, 5.5], [5.5, 5.5]]),
            atol=1e-12)
        np.testing.assert_allclose(
            aos_s[0][0], np.array([2.0, 90.0]), atol=1e-4)

    def test_vertex_pairing_property(self):
        """Column m of each input simplex must map onto column m of the
        paired output simplex (required by barycentric interpolation)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AIS, AOS = AIS2AOS_map(shower2x2,
                                   np.array([[1.0, 10.0], [1.0, 10.0]]),
                                   [3, 3], plot=False)
        ais_raw, aos_raw = _paired_simplices(AIS, AOS)
        for VA, VO in zip(ais_raw, aos_raw):
            for col in range(VA.shape[1]):
                np.testing.assert_allclose(shower2x2(VA[:, col]),
                                           VO[:, col], atol=1e-12)
