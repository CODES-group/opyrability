import numpy as np
import pytest
import polytope as pc

from opyrability import (
    are_overlapping,
    points2polyhedra,
    process_overlapping_polytopes,
    AIS2AOS_map,
)

# --------------------------------------------------------------------------- #
# Direct unit tests for the polytope geometry helpers that the rest of the
# suite only exercised indirectly. All expected values are computed by hand
# (box overlaps, union areas, grid-cell counts), independent of the code.
# --------------------------------------------------------------------------- #


class TestAreOverlapping:

    def test_overlapping_boxes(self):
        a = pc.box2poly(np.array([[0.0, 2.0], [0.0, 2.0]]))
        b = pc.box2poly(np.array([[1.0, 3.0], [1.0, 3.0]]))  # shares [1,2]^2
        assert are_overlapping(a, b) is True

    def test_disjoint_boxes(self):
        a = pc.box2poly(np.array([[0.0, 1.0], [0.0, 1.0]]))
        b = pc.box2poly(np.array([[5.0, 6.0], [5.0, 6.0]]))
        assert are_overlapping(a, b) is False


class TestProcessOverlappingPolytopes:
    """The de-overlap step must make the total volume equal the union area
    (no double counting), which is the whole point of the Vinson trick."""

    def test_overlapping_union_volume(self):
        a = pc.box2poly(np.array([[0.0, 2.0], [0.0, 1.0]]))  # area 2
        b = pc.box2poly(np.array([[1.0, 3.0], [0.0, 1.0]]))  # area 2
        # overlap [1,2]x[0,1] has area 1, so the union area is 2 + 2 - 1 = 3.
        bound_box = pc.box2poly(np.array([[0.0, 3.0], [0.0, 1.0]]))
        result = process_overlapping_polytopes(bound_box, pc.Region([a, b]))
        total = sum(p.volume for p in result)
        assert total == pytest.approx(3.0, abs=1e-6)

    def test_disjoint_volume_preserved(self):
        a = pc.box2poly(np.array([[0.0, 1.0], [0.0, 1.0]]))  # area 1
        b = pc.box2poly(np.array([[2.0, 3.0], [0.0, 1.0]]))  # area 1
        bound_box = pc.box2poly(np.array([[0.0, 3.0], [0.0, 1.0]]))
        result = process_overlapping_polytopes(bound_box, pc.Region([a, b]))
        total = sum(p.volume for p in result)
        assert total == pytest.approx(2.0, abs=1e-6)


class TestPoints2Polyhedra:
    """points2polyhedra tiles a gridded AIS/AOS into connected cells."""

    def _grids(self):
        def identity(u):
            return u
        bounds = np.array([[0.0, 2.0], [0.0, 2.0]])
        # AIS2AOS_map returns grid-shaped arrays, fed straight to the helper
        # exactly as multimodel_rep does.
        AIS, AOS = AIS2AOS_map(identity, bounds, [3, 3], plot=False)
        return AIS, AOS

    def test_cell_count_for_3x3_grid(self):
        AIS, AOS = self._grids()
        AIS_poly, AOS_poly = points2polyhedra(AIS, AOS)
        # A 3x3 grid has 2x2 = 4 connected cells.
        assert len(AOS_poly) == 4
        assert len(AIS_poly) == 4

    def test_identity_maps_inputs_to_equal_outputs(self):
        AIS, AOS = self._grids()
        AIS_poly, AOS_poly = points2polyhedra(AIS, AOS)
        # For y = u the AOS vertices of each cell equal its AIS vertices.
        for v_in, v_out in zip(AIS_poly, AOS_poly):
            a = v_in[np.lexsort(v_in.T)]
            b = v_out[np.lexsort(v_out.T)]
            np.testing.assert_allclose(a, b, atol=1e-9)

    def test_cells_tile_the_square(self):
        AIS, AOS = self._grids()
        _, AOS_poly = points2polyhedra(AIS, AOS)
        polys = [pc.qhull(v) for v in AOS_poly]
        # Each interior cell center lies in exactly one cell of the tiling.
        for pt in ([0.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.5, 0.5]):
            hits = sum(p.contains(np.array(pt).reshape(-1, 1)) for p in polys)
            assert hits == 1
