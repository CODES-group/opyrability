import pytest
import numpy as np
from opyrability import multimodel_rep, OI_eval, nlp_based_approach
from shower import shower2x2
from dma_mr import dma_mr_design

# -----------------------------------------------------------------------------
# Integration-level tests for opyrability.
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2023)
#
# The shower OI cases (forward and analytic-inverse) live in test_oi_eval.py
# with analytic anchors (OI=0 disjoint, OI=100 containment); they are not
# duplicated here. This file keeps a DMA-MR end-to-end check and a forward-
# consistency check of the NLP inverse map.
# -----------------------------------------------------------------------------

plot_flag = False
abs_tol = 1e-7
rel_tol = 1e-7


def test_dma_mr_design():
    """DMA-MR forward map: OI regression anchor plus an independent check
    that the AOS region actually encloses a known reachable output.

    The DMA-MR OI has no closed form, so the OI value is a regression anchor
    (it pins the geometry against drift). The containment assertion is the
    independent part: the achievable set must contain the output the model
    produces at an AIS grid point.
    """
    DOS_bounds = np.array([[20, 25],
                           [35, 45]])
    AIS_bounds = np.array([[10, 150],
                           [0.5, 2]])
    AIS_resolution = [5, 5]

    AOS_region = multimodel_rep(dma_mr_design,
                                AIS_bounds,
                                AIS_resolution,
                                plot=plot_flag)

    OI = OI_eval(AOS_region,
                 DOS_bounds,
                 plot=plot_flag)

    # Regression anchor (no analytic value exists for the DMA-MR OI).
    assert OI == pytest.approx(23.374694036948025, abs=1e-3, rel=1e-3)

    # Independent grounding: the AOS must contain the model output at an
    # interior AIS grid point ([80, 1.25] is on the [5, 5] grid).
    y_known = np.asarray(dma_mr_design(np.array([80.0, 1.25])), dtype=float)
    region = AOS_region[0]
    contained = any(p.contains(y_known.reshape(-1, 1)) for p in region)
    assert contained


def test_shower_inverse_nlp_forward_consistent():
    """The NLP inverse map must return inputs that actually produce the
    reported outputs: shower2x2(fDIS[i]) == fDOS[i] for every grid point.

    This replaces an earlier assertion that pinned only the L2 norm of the
    whole solution array (a lossy aggregate that many wrong solutions share).
    """
    u0 = np.array([0.0, 10.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([100.0, 100.0])

    DOS_bound = np.array([[17.5, 21.0],
                          [80.0, 100.0]])
    DOSresolution = [5, 5]

    fDIS, fDOS, message = nlp_based_approach(shower2x2,
                                             DOS_bound,
                                             DOSresolution,
                                             u0,
                                             lb,
                                             ub,
                                             method='ipopt',
                                             plot=plot_flag,
                                             ad=False,
                                             warmstart=True)

    for i in range(fDIS.shape[0]):
        np.testing.assert_allclose(shower2x2(fDIS[i]), fDOS[i], atol=1e-4)


if __name__ == '__main__':
    pytest.main()
