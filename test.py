"""Regression checks for the x=0 initial-plane construction."""

from __future__ import annotations

import unittest

import sympy as sp

from helicoid_plane_intersection import (
    build_geometry,
    derive_intersection,
    rotation_y,
    rotation_z,
    translation,
    verify,
)
from visualize_curve_in_initial_plane import (
    PlotParameters as CurvePlotParameters,
    transform_curve_to_initial_plane,
)
from visualize_geometry import PlotParameters as GeometryPlotParameters


class InitialPlaneXZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = build_geometry()
        cls.intersection = derive_intersection(cls.geometry)
        cls.flattened = transform_curve_to_initial_plane(
            cls.geometry, cls.intersection
        )

    def test_symbolic_geometry_is_consistent(self) -> None:
        verify(self.geometry, self.intersection)

    def test_plane_starts_from_x_zero(self) -> None:
        g = self.geometry
        expected_plane = sp.Matrix(
            [
                -g.v * sp.sin(g.gamma) * sp.cos(g.lam)
                + g.s * sp.sin(g.lam),
                g.r + g.v * sp.cos(g.gamma),
                g.v * sp.sin(g.gamma) * sp.sin(g.lam)
                + g.s * sp.cos(g.lam),
                1,
            ]
        )
        self.assertEqual(
            sp.simplify(g.plane_surface - expected_plane), sp.zeros(4, 1)
        )

    def test_inverse_transformed_curve_lies_in_x_zero(self) -> None:
        self.assertEqual(
            sp.trigsimp(self.flattened.homogeneous_curve[0]), 0
        )

    def test_forward_transform_restores_world_curve(self) -> None:
        g = self.geometry
        world_curve = sp.Matrix.vstack(self.intersection.curve, sp.Matrix([1]))
        forward_transform = (
            translation(dy=g.r) * rotation_y(g.lam) * rotation_z(g.gamma)
        )
        restored_curve = forward_transform * self.flattened.homogeneous_curve
        self.assertTrue(
            all(
                sp.trigsimp(sp.cancel(actual - expected)) == 0
                for actual, expected in zip(restored_curve, world_curve)
            )
        )

    def test_requested_default_parameters(self) -> None:
        expected = (2.536, 8.0, 15.0, 30.0, 10.0)
        for parameters in (CurvePlotParameters(), GeometryPlotParameters()):
            actual = (
                parameters.r,
                parameters.p,
                parameters.phi_deg,
                parameters.gamma_deg,
                parameters.lambda_deg,
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
