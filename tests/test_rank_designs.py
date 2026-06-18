import pytest
import numpy as np

from opyrability import rank_designs
from shower import shower2x2

# --------------------------------------------------------------------------- #
# Tests for rank_designs: ranking process designs by their Operability Index.
#
# The shower model gives fast, deterministic polytopic regions. The output
# perspective is checked against the known shower OI; a narrower input envelope
# is a subset of a wider one, so its AOS (and therefore its OI against a shared
# DOS) cannot exceed the wider envelope's, which fixes the ranking order.
# --------------------------------------------------------------------------- #

DOS = np.array([[10, 20], [70, 100]])
WIDE = np.array([[1, 10], [1, 10]])
NARROW = np.array([[3, 8], [3, 8]])
SHOWER_OI_WIDE = 60.23795007653283   # known OI_eval value for WIDE/DOS, [5, 5]


class TestRankDesignsOutputs:
    """Output-perspective ranking of two shower design envelopes."""

    def test_orders_by_OI(self):
        ranking = rank_designs({'wide': shower2x2, 'narrow': shower2x2},
                               AIS_bound={'wide': WIDE, 'narrow': NARROW},
                               DOS_bound=DOS,
                               resolution=[5, 5],
                               perspective='outputs',
                               plot=False)

        # Ranking is sorted from most to least operable.
        ois = [entry['OI'] for entry in ranking]
        assert ois == sorted(ois, reverse=True)

        # The wider input envelope is the more operable design and reproduces
        # the known shower OI; the narrow (subset) envelope cannot exceed it.
        wide = next(e for e in ranking if e['label'] == 'wide')
        narrow = next(e for e in ranking if e['label'] == 'narrow')
        assert ranking[0]['label'] == 'wide'
        assert wide['OI'] == pytest.approx(SHOWER_OI_WIDE, abs=1e-7, rel=1e-7)
        assert narrow['OI'] <= wide['OI']

        # Each entry carries the scored region.
        assert all('region' in e for e in ranking)


class TestRankDesignsInputs:
    """Input-perspective ranking runs (non-interactively) and is well-formed."""

    def test_runs_and_orders(self):
        ranking = rank_designs({'wide': shower2x2, 'narrow': shower2x2},
                               AIS_bound={'wide': WIDE, 'narrow': NARROW},
                               DOS_bound=DOS,
                               resolution=[4, 4],
                               perspective='inputs',
                               method='pounce',
                               plot=False)

        ois = [entry['OI'] for entry in ranking]
        assert len(ranking) == 2
        assert ois == sorted(ois, reverse=True)
        assert all(0.0 <= entry['OI'] <= 100.0 for entry in ranking)


class TestRankDesignsListInput:
    """A plain list of models is accepted and auto-labeled."""

    def test_accepts_list_and_autolabels(self):
        ranking = rank_designs([shower2x2, shower2x2],
                               AIS_bound=WIDE,
                               DOS_bound=DOS,
                               resolution=[5, 5],
                               plot=False)

        assert {e['label'] for e in ranking} == {'Design 1', 'Design 2'}
        # Identical models on a shared AIS give identical OIs.
        assert ranking[0]['OI'] == pytest.approx(ranking[1]['OI'])
