"""
Artifact tests for every opyrability plotting function.

These tests assert that the plotting functions actually produce a figure with
drawn content, exercise BOTH the matplotlib and plotly engines, and cover the
``plot=True`` side-effect paths (which the rest of the suite runs with
``plot=False``). They are deliberately regression tests for two classes of bug
that previously slipped through:

* a plotting helper crashing on a removed matplotlib API (e.g.
  ``matplotlib.cm.get_cmap``), caught here by running every ``plot=True`` path;
* a plotly figure being built but never displayed, caught here by asserting
  that ``_show_if_notebook`` is invoked for the plotly engine.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest

import opyrability
from opyrability import (
    dynamic_operability,
    dynamic_operability_mapping,
    dynamic_operability_scenarios,
    dOI_eval,
    plot_dynamic_funnel,
    plot_state_funnel,
    plot_funnel_comparison,
    simulate_mc_trajectories,
    AIS2AOS_map,
    multimodel_rep,
    OI_eval,
    nlp_based_approach,
    milp_based_approach,
)
from shower import shower2x2


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _close_figs():
    """Start and end each test with no open matplotlib figures."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def show_counter(monkeypatch):
    """Record every call to opyrability._show_if_notebook (the plotly display
    hook), so tests can assert a plotly figure is actually displayed."""
    calls = []
    monkeypatch.setattr(opyrability, '_show_if_notebook',
                        lambda fig: calls.append(fig))
    return calls


def _fig_has_artists(fig):
    """True if any axes of a matplotlib figure carries drawn content."""
    return any(ax.lines or ax.patches or ax.collections or ax.images
               for ax in fig.axes)


def _a_matplotlib_figure_was_drawn():
    """Assert at least one matplotlib figure with content currently exists."""
    nums = plt.get_fignums()
    assert nums, "no matplotlib figure was created"
    assert any(_fig_has_artists(plt.figure(n)) for n in nums), \
        "matplotlib figure(s) created but none carry drawn content"


# -- toy models -- #
def integrator_step(x, u):
    """x(k+1) = x(k) + u(k); identity output."""
    x_next = np.asarray(x, dtype=float) + np.asarray(u, dtype=float)
    return x_next, x_next


LTI = {'A': 0.9 * np.eye(2), 'B': np.eye(2), 'C': np.eye(2)}
AIS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
DOS = np.array([[-2.0, 2.0], [-2.0, 2.0]])


# --------------------------------------------------------------------------- #
# Dynamic plotters that return a figure
# --------------------------------------------------------------------------- #
class TestDynamicFunnelPlot:

    def _mapping(self):
        return dynamic_operability_mapping(
            integrator_step, np.zeros(2), AIS, 3, 4, plot=False)

    def test_matplotlib_artifact(self):
        res = self._mapping()
        dOI = dOI_eval(res, DOS, plot=False)
        fig, ax = plot_dynamic_funnel(res, DOS=DOS, dOI=dOI)
        assert ax is not None
        assert _fig_has_artists(fig)

    def test_plotly_artifact_and_display(self, show_counter):
        # Regression: the plotly funnel must be built AND displayed.
        pytest.importorskip('plotly')
        res = self._mapping()
        dOI = dOI_eval(res, DOS, plot=False)
        fig, ax = plot_dynamic_funnel(res, DOS=DOS, dOI=dOI, engine='plotly')
        assert ax is None
        assert len(fig.data) > 0
        assert len(show_counter) == 1, "plotly funnel was not displayed once"

    def test_plotly_show_false_suppresses_display(self, show_counter):
        pytest.importorskip('plotly')
        res = self._mapping()
        plot_dynamic_funnel(res, DOS=DOS, engine='plotly', show=False)
        assert len(show_counter) == 0


class TestStateFunnelPlot:

    def _projection(self, x0):
        return dynamic_operability(LTI, x0, AIS, k_max=4, plot=False)

    def test_matplotlib_artifact(self):
        fig, ax = plot_state_funnel(self._projection(np.zeros(2)))
        assert ax is not None and _fig_has_artists(fig)

    def test_plotly_artifact_and_display(self, show_counter):
        pytest.importorskip('plotly')
        fig, ax = plot_state_funnel(self._projection(np.zeros(2)),
                                    engine='plotly')
        assert len(fig.data) > 0
        assert len(show_counter) == 1


class TestFunnelComparisonPlot:

    def _two(self):
        a = dynamic_operability(LTI, np.zeros(2), AIS, k_max=4, plot=False)
        b = dynamic_operability(LTI, np.array([0.5, -0.5]), AIS, k_max=4,
                                plot=False)
        return {'A': a, 'B': b}

    def test_matplotlib_artifact(self):
        fig, ax = plot_funnel_comparison(self._two())
        assert _fig_has_artists(fig)

    def test_plotly_artifact_and_display(self, show_counter):
        pytest.importorskip('plotly')
        fig, ax = plot_funnel_comparison(self._two(), engine='plotly')
        assert len(fig.data) > 0
        assert len(show_counter) == 1


# --------------------------------------------------------------------------- #
# Side-effect plotting paths (plot=True) of the dynamic functions
# --------------------------------------------------------------------------- #
class TestDynamicPlotTrue:

    def test_mapping_plot_true_draws(self):
        # Regression for the removed matplotlib get_cmap API: this path was
        # never exercised because every other test used plot=False.
        dynamic_operability_mapping(
            integrator_step, np.zeros(2), AIS, 3, 4, plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_dOI_eval_plot_true_draws(self):
        res = dynamic_operability_mapping(
            integrator_step, np.zeros(2), AIS, 3, 4, plot=False)
        plt.close('all')
        dOI_eval(res, DOS, plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_dynamic_operability_plot_true_matplotlib(self):
        res = dynamic_operability(LTI, np.zeros(2), AIS, DOS=DOS, k_max=4,
                                  plot=True, engine='matplotlib')
        assert res['fig'] is not None and _fig_has_artists(res['fig'])

    def test_dynamic_operability_plot_true_plotly_displays_once(
            self, show_counter):
        # Regression: high-level plotly path must display exactly once
        # (not zero, and not twice now that plot_dynamic_funnel auto-shows).
        pytest.importorskip('plotly')
        res = dynamic_operability(LTI, np.zeros(2), AIS, DOS=DOS, k_max=4,
                                  monte_carlo=10, seed=0, plot=True,
                                  engine='plotly')
        assert len(res['fig'].data) > 0
        assert len(show_counter) == 1

    def test_scenarios_plot_true_matplotlib(self):
        def make_step(d):
            return integrator_step

        def make_x0(d):
            return np.zeros(2)

        dynamic_operability_scenarios(
            make_step, make_x0, AIS, scenarios={'a': 0.0, 'b': 1.0},
            DOS=DOS, k_max=4, method='nstep', plot=True,
            engine='matplotlib')
        _a_matplotlib_figure_was_drawn()

    def test_scenarios_plot_true_plotly_displays(self, show_counter):
        pytest.importorskip('plotly')

        def make_step(d):
            return integrator_step

        def make_x0(d):
            return np.zeros(2)

        dynamic_operability_scenarios(
            make_step, make_x0, AIS, scenarios={'a': 0.0, 'b': 1.0},
            DOS=DOS, k_max=4, method='nstep', plot=True, engine='plotly')
        assert len(show_counter) >= 1


# --------------------------------------------------------------------------- #
# Steady-state plotters (matplotlib), exercised through plot=True
# --------------------------------------------------------------------------- #
class TestSteadyStatePlotTrue:

    AIS_SS = np.array([[1.0, 10.0], [1.0, 10.0]])
    DOS_SS = np.array([[10.0, 20.0], [70.0, 100.0]])

    def test_AIS2AOS_map_plot_true_draws(self):
        AIS2AOS_map(shower2x2, self.AIS_SS, [3, 3], plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_multimodel_rep_plot_true_draws(self):
        multimodel_rep(shower2x2, self.AIS_SS, [3, 3], plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_OI_eval_plot_true_draws(self):
        region = multimodel_rep(shower2x2, self.AIS_SS, [4, 4], plot=False)
        plt.close('all')
        OI_eval(region, self.DOS_SS, plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_nlp_based_approach_plot_true_draws(self):
        nlp_based_approach(
            shower2x2, np.array([[10.0, 15.0], [80.0, 100.0]]), [2, 2],
            np.array([5.0, 5.0]), np.array([0.0, 0.0]),
            np.array([100.0, 100.0]), method='ipopt', ad=False, plot=True)
        _a_matplotlib_figure_was_drawn()

    def test_milp_based_approach_plot_true_draws(self):
        milp_based_approach(
            shower2x2, AIS_bound=np.array([[0.1, 10.0], [0.1, 10.0]]),
            PI_target=lambda u: u[0] + u[1],
            DOS_bounds=np.array([[6.0, 9.0], [85.0, 95.0]]),
            AIS_resolution=3,
            input_constr=(np.array([[1.0, -1.0]]), np.array([0.0])),
            plot=True)
        _a_matplotlib_figure_was_drawn()
