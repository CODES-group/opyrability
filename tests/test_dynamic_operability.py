import pytest
import numpy as np
import polytope as pc
from opyrability import (
    dynamic_operability,
    dynamic_operability_scenarios,
    dynamic_operability_mapping,
    dynamic_operability_nstep,
    dOI_eval,
    plot_dynamic_funnel,
    simulate_mc_trajectories,
    make_pyomo_step_model,
    propagate_output_covariance,
)
from opyrability import (
    _propagate_state_nonlinear,
    _dOI_at_step,
    _point_or_degenerate_polytope,
)

# --------------------------------------------------------------------------- #
# Dynamic operability tests -- mirror the steady-state (multimodel_rep,
# OI_eval) pair via (dynamic_operability_mapping, dOI_eval).
# --------------------------------------------------------------------------- #


# -- Simple step models (arity 2: step(x, u) -> (x_next, y)) -- #

def identity_step(x, u):
    """x(k+1) = x(k), ignores input. AOS should not grow."""
    x_next = np.asarray(x, dtype=float)
    return x_next, x_next


def integrator_step(x, u):
    """x(k+1) = x(k) + u(k). Simple integrator."""
    x_next = np.asarray(x, dtype=float) + np.asarray(u, dtype=float)
    return x_next, x_next


def linear_step_2d(x, u):
    """x(k+1) = 0.9 * x + u. Stable linear system."""
    x_next = 0.9 * np.asarray(x, dtype=float) + np.asarray(u, dtype=float)
    return x_next, x_next


def integrator_step_sum_output(x, u):
    """x(k+1) = x + u; y = [x[0] + x[1]]. Reduced scalar output."""
    x_next = np.asarray(x, dtype=float) + np.asarray(u, dtype=float)
    y = np.array([x_next[0] + x_next[1]])
    return x_next, y


# -- Arity-3 step models -- #

def integrator_step_with_d(x, u, d):
    """x(k+1) = x + u + d. Integrator with additive disturbance."""
    x_next = (np.asarray(x, dtype=float)
              + np.asarray(u, dtype=float)
              + np.asarray(d, dtype=float))
    return x_next, x_next


# --------------------------------------------------------------------------- #
# Mapping tests (nonlinear, arity 2)
# --------------------------------------------------------------------------- #


class TestMappingNonlinear:

    def test_identity_step_aos_bounded(self):
        """Identity step model: AOS volume should stay small."""
        x0 = np.array([1.0, 2.0])
        AIS_bound = np.array([[0.0, 1.0], [0.0, 1.0]])
        results = dynamic_operability_mapping(
            identity_step, x0, AIS_bound, 3, 5,
            convergence_tol=1e-10, plot=False,
        )
        assert 'AOS_regions' in results
        assert len(results['AOS_regions']) >= 1
        for v in results['volumes']:
            assert v < 1.0

    def test_integrator_aos_grows(self):
        """Integrator: AOS volume should be non-decreasing."""
        x0 = np.array([0.0, 0.0])
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 5, plot=False,
        )
        vols = results['volumes']
        for i in range(1, len(vols)):
            assert vols[i] >= vols[i - 1] - 1e-8

    def test_convergence_detected(self):
        """Stable linear system should trigger early stopping."""
        x0 = np.array([0.0, 0.0])
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            linear_step_2d, x0, AIS_bound, 3, 100,
            convergence_tol=1e-3, plot=False,
        )
        assert results['k_converged'] is not None
        assert results['k_converged'] < 100

    def test_output_shapes(self):
        """Result dict should have the documented structure."""
        x0 = np.array([0.0, 0.0])
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 3, plot=False,
        )
        assert len(results['AOS_x']) == 4  # k=0 initial + 3 steps
        assert len(results['AOS_regions']) == 3
        assert len(results['volumes']) == 3

    def test_reduced_output_dimension(self):
        """Step model returning a scalar output should yield 1D AOS."""
        x0 = np.array([0.0, 0.0])
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step_sum_output, x0, AIS_bound, 3, 3, plot=False,
        )
        verts = pc.extreme(results['AOS_regions'][-1].list_poly[0])
        assert verts is not None
        assert verts.shape[1] == 1

    def test_results_stores_system(self):
        """Private keys carry system definition for downstream use."""
        x0 = np.array([0.0, 0.0])
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 3, plot=False,
        )
        assert results['_step_model'] is integrator_step
        assert results['_arity'] == 2
        np.testing.assert_array_equal(results['_x0'], x0)
        np.testing.assert_array_equal(results['_AIS_bound'], AIS_bound)
        assert results['_EDS_bound'] is None
        assert results['_matrices'] is None


# --------------------------------------------------------------------------- #
# Mapping tests (linear fast path via A/B/C)
# --------------------------------------------------------------------------- #


class TestMappingLinear:

    def test_linear_fast_path_runs(self):
        """Passing A, B, C should activate the Minkowski-sum path."""
        A = np.array([[0.9, 0.0], [0.0, 0.8]])
        B = np.eye(2)
        C = np.eye(2)
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])

        results = dynamic_operability_mapping(
            step_model=None, x0=x0, AIS_bound=AIS_bound,
            AIS_resolution=3, k_max=50,
            A=A, B=B, C=C,
            convergence_tol=1e-3, plot=False,
        )
        assert results['k_converged'] is not None
        assert results['_matrices'] is not None
        final_verts = pc.extreme(results['AOS_regions'][-1].list_poly[0])
        bbox_range = final_verts.max(axis=0) - final_verts.min(axis=0)
        assert np.all(bbox_range > 0.5)

    def test_linear_matches_nonlinear(self):
        """Linear fast path and nonlinear path should agree numerically."""
        A = np.array([[0.9, 0.0], [0.0, 0.8]])
        B = np.eye(2)
        C = np.eye(2)

        def nl_step(x, u):
            x_next = A @ np.asarray(x) + B @ np.asarray(u)
            y = C @ x_next
            return x_next, y

        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])

        res_lin = dynamic_operability_mapping(
            step_model=None, x0=x0, AIS_bound=AIS_bound,
            AIS_resolution=3, k_max=10,
            A=A, B=B, C=C, plot=False,
        )
        res_nl = dynamic_operability_mapping(
            step_model=nl_step, x0=x0, AIS_bound=AIS_bound,
            AIS_resolution=3, k_max=10, plot=False,
        )
        # For an LTI system both paths are exact, so they agree to machine
        # precision (measured ~4e-16); a tight rtol makes this a real check.
        np.testing.assert_allclose(
            res_lin['volumes'], res_nl['volumes'], rtol=1e-7,
        )

    def test_step_model_or_matrices_required(self):
        """Both step_model and matrices missing should raise."""
        with pytest.raises(ValueError, match="step_model"):
            dynamic_operability_mapping(
                step_model=None, x0=np.zeros(2),
                AIS_bound=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                AIS_resolution=3, k_max=3, plot=False,
            )


# --------------------------------------------------------------------------- #
# Arity and EDS validation
# --------------------------------------------------------------------------- #


class TestValidation:

    def test_arity_2_with_eds_raises(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        EDS_bound = np.array([[-0.1, 0.1], [-0.1, 0.1]])
        with pytest.raises(ValueError, match="EDS_bound"):
            dynamic_operability_mapping(
                integrator_step, x0, AIS_bound, 3, 3,
                EDS_bound=EDS_bound, plot=False,
            )

    def test_arity_3_without_eds_raises(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        with pytest.raises(ValueError, match="EDS_bound"):
            dynamic_operability_mapping(
                integrator_step_with_d, x0, AIS_bound, 3, 3,
                EDS_bound=None, plot=False,
            )

    def test_missing_x0_raises(self):
        with pytest.raises(ValueError, match="x0"):
            dynamic_operability_mapping(
                integrator_step, None,
                np.array([[-1.0, 1.0], [-1.0, 1.0]]), 3, 3,
                plot=False,
            )


# --------------------------------------------------------------------------- #
# Disturbance handling (arity 3)
# --------------------------------------------------------------------------- #


class TestDisturbances:

    def test_disturbance_enlarges_aos(self):
        """Achievable-set semantics: AOS under disturbances should be
        at least as large as the no-disturbance AOS, since disturbances
        add reachable states via the Minkowski sum B_d @ EDS."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        EDS_bound = np.array([[-0.5, 0.5], [-0.5, 0.5]])

        res_with_d = dynamic_operability_mapping(
            integrator_step_with_d, x0, AIS_bound, 3, 3,
            EDS_bound=EDS_bound, plot=False,
        )

        # Baseline: same dynamics but d fixed to zero, arity 2.
        def step_d_zero(x, u):
            x_next = (np.asarray(x, dtype=float)
                      + np.asarray(u, dtype=float))
            return x_next, x_next

        res_no_d = dynamic_operability_mapping(
            step_d_zero, x0, AIS_bound, 3, 3, plot=False,
        )
        for v_no, v_with in zip(res_no_d['volumes'],
                                 res_with_d['volumes']):
            assert v_with >= v_no - 1e-4


# --------------------------------------------------------------------------- #
# dOI_eval tests
# --------------------------------------------------------------------------- #


class TestdOIEval:

    def test_doi_in_valid_range(self):
        """DOI values must lie in [0, 100]."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        DOS = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        results = dynamic_operability_mapping(
            linear_step_2d, x0, AIS_bound, 3, 10, plot=False,
        )
        dOI = dOI_eval(results, DOS, plot=False)
        assert dOI.shape == (len(results['AOS_regions']),)
        for value in dOI:
            assert 0.0 <= value <= 100.0 + 1e-6

    def test_doi_length_matches_k_max(self):
        """dOI array length matches the mapping horizon."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        DOS = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 4, plot=False,
        )
        dOI = dOI_eval(results, DOS, plot=False)
        assert len(dOI) == len(results['AOS_regions'])


# --------------------------------------------------------------------------- #
# simulate_mc_trajectories tests
# --------------------------------------------------------------------------- #


class TestSimulateMC:

    def test_mc_shape_arity2(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 5, plot=False,
        )
        trajs = simulate_mc_trajectories(results,
                                         n_trajectories=10, seed=42)
        assert trajs.shape == (10, 6, 2)

    def test_mc_n_steps_override(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 5, plot=False,
        )
        trajs = simulate_mc_trajectories(results, n_steps=3,
                                         n_trajectories=10, seed=0)
        assert trajs.shape == (10, 4, 2)

    def test_mc_seed_reproducibility(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_mapping(
            integrator_step, x0, AIS_bound, 3, 3, plot=False,
        )
        t1 = simulate_mc_trajectories(results, n_trajectories=5, seed=7)
        t2 = simulate_mc_trajectories(results, n_trajectories=5, seed=7)
        np.testing.assert_array_equal(t1, t2)

    def test_mc_arity_3(self):
        """MC simulation must work when step_model has arity 3."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        EDS_bound = np.array([[-0.1, 0.1], [-0.1, 0.1]])
        results = dynamic_operability_mapping(
            integrator_step_with_d, x0, AIS_bound, 3, 3,
            EDS_bound=EDS_bound, plot=False,
        )
        trajs = simulate_mc_trajectories(results,
                                         n_trajectories=4, seed=0)
        assert trajs.shape == (4, 4, 2)


# --------------------------------------------------------------------------- #
# dynamic_operability_nstep tests -- n-step output-space funnel for
# (possibly high-dimensional) nonlinear step models.
# --------------------------------------------------------------------------- #


class TestNStep:

    def test_returns_regions_and_volumes(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_nstep(
            integrator_step, x0, AIS_bound, 4, AIS_resolution=3, plot=False,
        )
        assert len(results['AOS_regions']) == 4
        assert len(results['volumes']) == 4
        # Private keys must be present for the downstream functions.
        for key in ('_step_model', '_arity', '_x0', '_AIS_bound',
                    '_EDS_bound'):
            assert key in results

    def test_integrator_funnel_grows(self):
        """Integrator AOS area must be non-decreasing as the funnel expands."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_nstep(
            integrator_step, x0, AIS_bound, 4, AIS_resolution=3, plot=False,
        )
        vols = results['volumes']
        assert np.all(np.diff(vols) >= -1e-9)
        assert vols[-1] > vols[0]

    def test_identity_funnel_stays_point(self):
        """Identity step never moves the state: AOS area stays ~0."""
        x0 = np.array([1.0, 2.0])
        AIS_bound = np.array([[0.0, 1.0], [0.0, 1.0]])
        results = dynamic_operability_nstep(
            identity_step, x0, AIS_bound, 3, AIS_resolution=3, plot=False,
        )
        assert np.all(results['volumes'] < 1e-6)

    def test_y0_prepends_initial_point(self):
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        y0 = np.zeros(2)
        results = dynamic_operability_nstep(
            integrator_step, x0, AIS_bound, 3, AIS_resolution=3,
            y0=y0, plot=False,
        )
        # k=0 slice is the degenerate steady-state point.
        assert len(results['AOS_regions']) == 4
        assert results['volumes'][0] == pytest.approx(0.0, abs=1e-9)

    def test_results_chain_with_dOI_and_mc(self):
        """nstep results must work directly with dOI_eval and MC sampling."""
        x0 = np.zeros(2)
        AIS_bound = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        results = dynamic_operability_nstep(
            integrator_step, x0, AIS_bound, 3, AIS_resolution=3, plot=False,
        )
        DOS = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        dOI = dOI_eval(results, DOS, plot=False)
        assert len(dOI) == len(results['AOS_regions'])
        assert np.all((dOI >= 0.0) & (dOI <= 100.0))
        trajs = simulate_mc_trajectories(results, n_trajectories=5, seed=0)
        assert trajs.shape == (5, 4, 2)


# --------------------------------------------------------------------------- #
# High-level API: dynamic_operability and dynamic_operability_scenarios.
# --------------------------------------------------------------------------- #


class TestHighLevelAPI:

    AIS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    DOS = np.array([[-2.0, 2.0], [-2.0, 2.0]])

    def test_auto_projection_for_small_state(self):
        res = dynamic_operability(
            integrator_step, np.zeros(2), self.AIS, k_max=4, plot=False,
        )
        assert 'projection' in res['method']
        assert len(res['AOS_regions']) >= 1

    def test_force_nstep(self):
        res = dynamic_operability(
            integrator_step, np.zeros(2), self.AIS, k_max=4,
            method='nstep', plot=False,
        )
        assert res['method'] == 'n-step simulation'

    def test_matrix_model_is_linear(self):
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res = dynamic_operability(
            model, np.zeros(2), self.AIS, k_max=4, plot=False,
        )
        assert 'linear' in res['method']

    def test_matrix_model_with_refs_is_pure_shift(self):
        """u_ref/y_ref let deviation models use absolute AIS/DOS: the
        funnel must be the deviation funnel translated by y_ref."""
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res_dev = dynamic_operability(
            model, np.zeros(2), self.AIS, DOS=self.DOS, k_max=4,
            plot=False,
        )
        u_ref = np.array([100.0, 200.0])
        y_ref = np.array([50.0, -30.0])
        model_abs = dict(model, u_ref=u_ref, y_ref=y_ref)
        AIS_abs = self.AIS + u_ref[:, None]
        DOS_abs = self.DOS + y_ref[:, None]
        res_abs = dynamic_operability(
            model_abs, np.zeros(2), AIS_abs, DOS=DOS_abs, k_max=4,
            plot=False,
        )
        np.testing.assert_allclose(res_abs['dOI'], res_dev['dOI'],
                                   atol=1e-6)
        for r_dev, r_abs in zip(res_dev['AOS_regions'],
                                res_abs['AOS_regions']):
            p_dev = r_dev.list_poly[0]
            p_abs = r_abs.list_poly[0]
            # Same polytope translated by y_ref: equal volume and every
            # shifted deviation vertex contained in the absolute polytope.
            assert p_abs.volume == pytest.approx(p_dev.volume, abs=1e-6)
            for v in pc.extreme(p_dev) + y_ref:
                assert p_abs.contains(v.reshape(-1, 1), abs_tol=1e-6)

    def test_refs_rejected_for_callable_model(self):
        with pytest.raises(ValueError):
            dynamic_operability_mapping(
                integrator_step, np.zeros(2), self.AIS, 3, 3,
                u_ref=np.zeros(2), plot=False,
            )

    def test_dOI_attached_and_ranged(self):
        res = dynamic_operability(
            integrator_step, np.zeros(2), self.AIS, DOS=self.DOS,
            k_max=4, plot=False,
        )
        assert 'dOI' in res
        assert len(res['dOI']) == len(res['AOS_regions'])
        assert np.all((res['dOI'] >= 0.0) & (res['dOI'] <= 100.0))

    def test_no_plot_no_fig_key(self):
        res = dynamic_operability(
            integrator_step, np.zeros(2), self.AIS, k_max=3, plot=False,
        )
        assert 'fig' not in res

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            dynamic_operability(
                integrator_step, np.zeros(2), self.AIS, k_max=2,
                method='bogus', plot=False,
            )

    def test_funnel_plot_with_dOI_runs(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        res = dynamic_operability_mapping(
            integrator_step, np.zeros(2), self.AIS, 3, 4, plot=False)
        dOI = dOI_eval(res, self.DOS, plot=False)
        fig, ax = plot_dynamic_funnel(res, DOS=self.DOS, dOI=dOI)
        assert fig is not None
        plt.close(fig)

    def test_funnel_plot_plotly_engine(self):
        plotly = pytest.importorskip('plotly')
        res = dynamic_operability_mapping(
            integrator_step, np.zeros(2), self.AIS, 3, 4, plot=False)
        dOI = dOI_eval(res, self.DOS, plot=False)
        for orient in ('landscape', 'vertical'):
            fig, ax = plot_dynamic_funnel(
                res, DOS=self.DOS, dOI=dOI, engine='plotly',
                orientation=orient)
            assert ax is None
            assert len(fig.data) > 0
            assert {t.type for t in fig.data} <= {'mesh3d', 'scatter3d'}

    def test_funnel_plot_invalid_engine_raises(self):
        res = dynamic_operability_mapping(
            integrator_step, np.zeros(2), self.AIS, 3, 3, plot=False)
        with pytest.raises(ValueError):
            plot_dynamic_funnel(res, engine='bogus')

    def test_state_funnel_single_and_overlay(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from opyrability import plot_state_funnel
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res_a = dynamic_operability(model, np.zeros(2), self.AIS,
                                    k_max=4, plot=False)
        res_b = dynamic_operability(model, np.array([1.0, 1.0]),
                                    self.AIS, k_max=4, plot=False)
        fig, ax = plot_state_funnel(res_a)
        assert fig is not None and ax is not None
        plt.close(fig)
        fig, ax = plot_state_funnel({'nominal': res_a, 'updated': res_b})
        assert fig is not None
        plt.close(fig)

    def test_state_funnel_requires_projection_results(self):
        from opyrability import plot_state_funnel
        res = dynamic_operability(
            integrator_step, np.zeros(2), self.AIS, k_max=3,
            method='nstep', plot=False,
        )
        with pytest.raises(ValueError):
            plot_state_funnel(res)

    def test_update_dynamic_funnel_matches_recompute(self):
        from opyrability import update_dynamic_funnel
        model = {'A': 0.8 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res_off = dynamic_operability(model, np.zeros(2), self.AIS,
                                      DOS=self.DOS, k_max=5, plot=False)
        x0_new = np.array([0.7, -0.4])
        res_upd = update_dynamic_funnel(res_off, x0_new, DOS=self.DOS)
        res_rec = dynamic_operability(model, x0_new, self.AIS,
                                      DOS=self.DOS, k_max=5, plot=False)
        np.testing.assert_allclose(res_upd['dOI'], res_rec['dOI'],
                                   atol=1e-6)
        # Translation preserves volumes.
        np.testing.assert_allclose(
            [r.list_poly[0].volume for r in res_upd['AOS_regions']],
            [r.list_poly[0].volume for r in res_off['AOS_regions']],
            rtol=1e-6)

    def test_update_dynamic_funnel_rejects_nonlinear(self):
        from opyrability import update_dynamic_funnel
        res = dynamic_operability(integrator_step, np.zeros(2), self.AIS,
                                  k_max=3, method='nstep', plot=False)
        with pytest.raises(ValueError):
            update_dynamic_funnel(res, np.ones(2))

    def test_gaussian_robust_funnel_shrinks(self):
        from opyrability import (gaussian_robust_funnel,
                                 propagate_output_covariance)
        model = {'A': 0.8 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res = dynamic_operability(model, np.zeros(2), self.AIS,
                                  DOS=self.DOS, k_max=5, plot=False)
        Sy = propagate_output_covariance(0.5 * np.eye(2),
                                         0.1 * np.eye(2), np.eye(2),
                                         np.eye(2), 5)
        assert len(Sy) == 5 and Sy[0].shape == (2, 2)
        rob = gaussian_robust_funnel(res, Sy, confidence=0.95,
                                     DOS=self.DOS)
        # Shrunken slices can never exceed the nominal ones.
        assert np.all(rob['volumes'] <= res['volumes'] + 1e-9)
        assert np.all(rob['dOI'] <= res['dOI'] + 1e-9)

    def test_gaussian_robust_funnel_empties_under_huge_noise(self):
        from opyrability import gaussian_robust_funnel
        model = {'A': 0.8 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res = dynamic_operability(model, np.zeros(2), self.AIS,
                                  k_max=3, plot=False)
        rob = gaussian_robust_funnel(res, 1e6 * np.eye(2))
        assert np.all(rob['volumes'] == 0.0)

    def test_identify_lti_recovers_linear_system(self):
        from opyrability import identify_lti_step_tests
        A = np.diag([0.8, 0.6])
        B = np.array([[0.2, 0.0], [0.0, 0.4]])
        C = np.array([[1.0, 0.5], [0.0, 1.0]])
        u_ref = np.array([10.0, 20.0])
        y_ref = np.array([1.0, 2.0])

        def lti_step(x, u):
            xn = A @ np.asarray(x, float) + B @ (np.asarray(u, float)
                                                 - u_ref)
            return xn, C @ xn + y_ref

        ident = identify_lti_step_tests(lti_step, np.zeros(2), u_ref,
                                        du=1.0, n_steps=40)
        dc_true = C @ np.linalg.inv(np.eye(2) - A) @ B
        dc_id = (ident['C']
                 @ np.linalg.inv(np.eye(ident['A'].shape[0]) - ident['A'])
                 @ ident['B'])
        # 40-step responses settle this system fully; the identified DC gain
        # matches the analytic one to ~1e-4, so atol=1e-3 is a real check.
        np.testing.assert_allclose(dc_id, dc_true, atol=1e-3)
        np.testing.assert_allclose(ident['y_ref'], y_ref, atol=1e-6)

    def test_funnel_comparison_overlay(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from opyrability import plot_funnel_comparison
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res_lin = dynamic_operability(model, np.zeros(2), self.AIS,
                                      k_max=4, plot=False)
        res_ns = dynamic_operability(integrator_step, np.zeros(2),
                                     self.AIS, k_max=4, method='nstep',
                                     plot=False)
        fig, ax = plot_funnel_comparison(
            {'linear': res_lin, 'nstep': res_ns})
        assert fig is not None
        plt.close(fig)

    def test_funnel_comparison_plotly(self):
        plotly = pytest.importorskip('plotly')
        from opyrability import plot_funnel_comparison
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res = dynamic_operability(model, np.zeros(2), self.AIS,
                                  k_max=3, plot=False)
        fig, ax = plot_funnel_comparison({'a': res, 'b': res},
                                         engine='plotly')
        assert ax is None and len(fig.data) > 0

    def test_state_funnel_plotly(self):
        plotly = pytest.importorskip('plotly')
        from opyrability import plot_state_funnel
        model = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
        res = dynamic_operability(model, np.zeros(2), self.AIS,
                                  k_max=4, plot=False)
        fig, ax = plot_state_funnel(res, engine='plotly')
        assert ax is None and len(fig.data) > 0

    def test_scenarios_plotly_engine(self):
        plotly = pytest.importorskip('plotly')

        def step_factory(d):
            def step(x, u):
                xn = np.asarray(x, dtype=float) + np.asarray(u, dtype=float) + d
                return xn, xn
            return step

        res = dynamic_operability_scenarios(
            step_factory, lambda d: np.zeros(2), self.AIS,
            scenarios={'a': 0.0, 'b': 0.3}, DOS=self.DOS, k_max=3,
            engine='plotly',
        )
        assert res['ax'] is None
        assert len(res['fig'].data) > 0

    def test_scenarios_structure_and_intersection(self):
        def step_factory(d):
            def step(x, u):
                xn = np.asarray(x, dtype=float) + np.asarray(u, dtype=float) + d
                return xn, xn
            return step

        def x0_factory(d):
            return np.zeros(2)

        DOS = np.array([[-3.0, 3.0], [-3.0, 3.0]])
        res = dynamic_operability_scenarios(
            step_factory, x0_factory, self.AIS,
            scenarios={'a': 0.0, 'b': 0.3}, DOS=DOS, k_max=3, plot=False,
        )
        assert set(res['scenarios'].keys()) == {'a', 'b'}
        inter = res['intersection']
        assert len(inter['AOS_regions']) == 3
        assert len(inter['volumes']) == 3
        assert 'dOI' in inter and len(inter['dOI']) == 3


# --------------------------------------------------------------------------- #
# Private helper tests
# --------------------------------------------------------------------------- #


class TestHelpers:

    def test_propagate_state_nonlinear_shape(self):
        state_verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        input_verts = np.array([[-1.0, -1.0], [1.0, 1.0]])
        next_pts, out_pts = _propagate_state_nonlinear(
            integrator_step, state_verts, input_verts,
        )
        assert next_pts.shape == (6, 2)
        assert out_pts.shape == (6, 2)

    def test_propagate_with_disturbance(self):
        state_verts = np.array([[0.0, 0.0]])
        input_verts = np.array([[1.0, 1.0]])
        d = np.array([0.5, 0.5])
        next_pts, out_pts = _propagate_state_nonlinear(
            integrator_step_with_d, state_verts, input_verts, d_vec=d,
        )
        np.testing.assert_allclose(next_pts[0], [1.5, 1.5])
        np.testing.assert_allclose(out_pts[0], [1.5, 1.5])

    def test_point_or_degenerate_polytope(self):
        pt = np.array([[3.0, 4.0]])
        poly = _point_or_degenerate_polytope(pt)
        assert poly.dim == 2
        assert poly.contains(np.array([[3.0], [4.0]]))

    def test_dOI_at_step_contains(self):
        """dOI should be ~100% when AOS fully contains DOS."""
        big_poly = pc.box2poly(np.array([[-5.0, 5.0], [-5.0, 5.0]]))
        region = pc.Region([big_poly])
        DOS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        # Axis-aligned boxes give an exact intersection, so dOI is exactly 100.
        assert _dOI_at_step(region, DOS) == pytest.approx(100.0, abs=1e-9)

    def test_dOI_at_step_disjoint(self):
        """dOI should be ~0% when AOS and DOS do not overlap."""
        small_poly = pc.box2poly(np.array([[10.0, 11.0], [10.0, 11.0]]))
        region = pc.Region([small_poly])
        DOS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        # Disjoint boxes give an empty intersection, so dOI is exactly 0.
        assert _dOI_at_step(region, DOS) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Closed-form correctness of the Gaussian covariance propagation
# --------------------------------------------------------------------------- #


class TestPropagateOutputCovariance:
    """Check propagate_output_covariance against a hand-computed recursion.

    For a scalar system x(k+1) = a x(k) + w, y = x, with disturbance variance
    s, the recursion Sx(k+1) = a^2 Sx(k) + s and Sx(0) = 0 gives the closed
    form Sy(k) = s * sum_{j=0}^{k-1} a^(2j).
    """

    def test_scalar_geometric_series(self):
        a, s = 0.5, 2.0
        Sy = propagate_output_covariance(
            np.array([[a]]), np.array([[1.0]]), np.array([[1.0]]),
            np.array([[s]]), 3)
        vals = [float(np.asarray(S).ravel()[0]) for S in Sy]
        # s, s(1+a^2), s(1+a^2+a^4) = 2.0, 2.5, 2.625
        assert vals == pytest.approx([2.0, 2.5, 2.625], abs=1e-12)

    def test_output_matrix_scales_quadratically(self):
        # C scales the output covariance by C^2 for a scalar system.
        a, s, c = 0.5, 2.0, 3.0
        Sy = propagate_output_covariance(
            np.array([[a]]), np.array([[1.0]]), np.array([[c]]),
            np.array([[s]]), 2)
        vals = [float(np.asarray(S).ravel()[0]) for S in Sy]
        assert vals == pytest.approx([c**2 * 2.0, c**2 * 2.5], abs=1e-12)

    def test_measurement_noise_is_added(self):
        a, s, v = 0.5, 2.0, 7.0
        Sy = propagate_output_covariance(
            np.array([[a]]), np.array([[1.0]]), np.array([[1.0]]),
            np.array([[s]]), 2, Sigma_v=np.array([[v]]))
        vals = [float(np.asarray(S).ravel()[0]) for S in Sy]
        assert vals == pytest.approx([2.0 + v, 2.5 + v], abs=1e-12)


# --------------------------------------------------------------------------- #
# make_pyomo_step_model correctness against a known linear step
# --------------------------------------------------------------------------- #


class TestMakePyomoStepModel:
    """The wrapped Pyomo step must reproduce the analytic next state."""

    def test_linear_step_matches_analytic(self):
        pyo = pytest.importorskip("pyomo.environ")

        def build_linear():
            m = pyo.ConcreteModel()
            m.si = pyo.RangeSet(0, 1)
            m.ui = pyo.RangeSet(0, 1)
            m.x_current = pyo.Param(m.si, initialize=0.0, mutable=True)
            m.u = pyo.Var(m.ui, initialize=0.0)
            m.x_next = pyo.Var(m.si, initialize=0.0)

            @m.Constraint(m.si)
            def dyn(m, i):
                # x_next = 0.5 * x_current + u
                return m.x_next[i] == 0.5 * m.x_current[i] + m.u[i]
            m.obj = pyo.Objective(expr=0)
            return m

        step = make_pyomo_step_model(build_linear, n_x=2, n_u=2)
        x_next, y = step(np.array([2.0, 4.0]), np.array([1.0, 1.0]))
        # 0.5 * [2, 4] + [1, 1] = [2, 3]
        np.testing.assert_allclose(x_next, [2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(y, x_next, atol=1e-12)
