"""Transform the intersection curve back to x'=0 and draw it in 2D."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from helicoid_plane_intersection import (
    Geometry,
    Intersection,
    build_geometry,
    derive_intersection,
    rotation_y,
    rotation_z,
    translation,
)


@dataclass(frozen=True)
class FlattenedCurve:
    """Curve expressed in coordinates of the initial plane x'=0."""

    inverse_transform: sp.Matrix
    homogeneous_curve: sp.Matrix
    curve_yz: sp.Matrix


@dataclass(frozen=True)
class PlotParameters:
    """Numeric parameters and displayed part of the curve."""

    r: float = 2.536
    p: float = 8.0
    phi_deg: float = 15.0
    gamma_deg: float = 30.0
    lambda_deg: float = 10.0
    u_min: float = -2.5
    u_max: float = 4.0
    turns: float = 2.0
    samples: int = 1600


def transform_curve_to_initial_plane(
    geometry: Geometry, intersection: Intersection
) -> FlattenedCurve:
    """Apply the inverse plane transform and retain the local y', z' pair.

    The forward plane transform is

        T_y(r) * R_y(lambda) * R_z(gamma).

    Consequently, its inverse is

        R_z(-gamma) * R_y(-lambda) * T_y(-r).
    """
    inverse_transform = (
        rotation_z(-geometry.gamma)
        * rotation_y(-geometry.lam)
        * translation(dy=-geometry.r)
    )
    world_curve_homogeneous = sp.Matrix.vstack(
        intersection.curve, sp.Matrix([1])
    )
    local_curve = inverse_transform * world_curve_homogeneous

    local_x = sp.trigsimp(sp.cancel(local_curve[0]))
    if local_x != 0:
        raise AssertionError(
            "inverse-transformed curve does not satisfy x' = 0"
        )

    homogeneous_curve = sp.Matrix(
        [
            sp.Integer(0),
            sp.cancel(local_curve[1]),
            sp.cancel(local_curve[2]),
            sp.Integer(1),
        ]
    )
    curve_yz = sp.Matrix([homogeneous_curve[1], homogeneous_curve[2]])
    print(sp.latex( sp.trigsimp(homogeneous_curve[1])) )
    print(sp.latex( sp.trigsimp(homogeneous_curve[2])) )

    return FlattenedCurve(
        inverse_transform=inverse_transform,
        homogeneous_curve=homogeneous_curve,
        curve_yz=curve_yz,
    )


def _vector_function(
    expression: sp.Matrix, argument: sp.Symbol
):
    """Lambdify a vector while broadcasting constant components."""
    component_functions = [
        sp.lambdify(argument, expression[index], modules="numpy", cse=True)
        for index in range(expression.rows)
    ]

    def evaluate(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.broadcast_to(
                    np.asarray(function(values), dtype=float), values.shape
                )
                for function in component_functions
            ],
            axis=0,
        )

    return evaluate


def render_flattened_curve(
    parameters: PlotParameters,
    output_path: Path,
    *,
    dpi: int = 180,
    show: bool = False,
) -> Path:
    """Create the y'z' plot of the inverse-transformed curve."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    geometry = build_geometry()
    intersection = derive_intersection(geometry)
    flattened = transform_curve_to_initial_plane(geometry, intersection)

    substitutions = {
        geometry.r: parameters.r,
        geometry.p: parameters.p,
        geometry.phi: np.deg2rad(parameters.phi_deg),
        geometry.gamma: np.deg2rad(parameters.gamma_deg),
        geometry.lam: np.deg2rad(parameters.lambda_deg),
    }
    theta_limit = parameters.turns * np.pi
    theta_values = np.linspace(-theta_limit, theta_limit, parameters.samples)
    #theta_values = np.linspace(-3, 0, parameters.samples)

    curve_function = _vector_function(
        flattened.curve_yz.subs(substitutions), geometry.theta
    )
    curve_yz = curve_function(theta_values)

    u_function = sp.lambdify(
        geometry.theta,
        intersection.u_of_theta.subs(substitutions),
        modules="numpy",
        cse=True,
    )
    denominator_function = sp.lambdify(
        geometry.theta,
        intersection.D.subs(substitutions),
        modules="numpy",
        cse=True,
    )
    curve_u = np.asarray(u_function(theta_values), dtype=float)
    denominator = np.asarray(denominator_function(theta_values), dtype=float)
    denominator_scale = max(float(np.nanmax(np.abs(denominator))), 1.0)
    valid_points = (
        np.all(np.isfinite(curve_yz), axis=0)
        & np.isfinite(curve_u)
        & (np.abs(denominator) > 1e-7 * denominator_scale)
        & (curve_u >= parameters.u_min)
        & (curve_u <= parameters.u_max)
    )

    points = curve_yz.T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    valid_segments = valid_points[:-1] & valid_points[1:]
    segment_theta = (theta_values[:-1] + theta_values[1:]) / 2

    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    normalization = Normalize(vmin=-theta_limit, vmax=theta_limit)
    collection = LineCollection(
        segments[valid_segments],
        array=segment_theta[valid_segments],
        cmap="viridis",
        norm=normalization,
        linewidth=2.8,
    )
    ax.add_collection(collection)

    valid_indices = np.flatnonzero(valid_points)
    if valid_indices.size == 0:
        raise ValueError(
            "no regular curve points fall inside the selected u range"
        )
    start_index = valid_indices[0]
    end_index = valid_indices[-1]
    ax.scatter(
        curve_yz[0, start_index],
        curve_yz[1, start_index],
        color="#2563a8",
        edgecolor="white",
        linewidth=0.8,
        s=58,
        zorder=5,
        label=rf"$\theta={theta_values[start_index]:.2f}$",
    )
    ax.scatter(
        curve_yz[0, end_index],
        curve_yz[1, end_index],
        color="#b42318",
        edgecolor="white",
        linewidth=0.8,
        marker="s",
        s=52,
        zorder=5,
        label=rf"$\theta={theta_values[end_index]:.2f}$",
    )

    valid_coordinates = curve_yz[:, valid_points]
    y_min, z_min = np.min(valid_coordinates, axis=1)
    y_max, z_max = np.max(valid_coordinates, axis=1)
    span = max(y_max - y_min, z_max - z_min, 1.0)
    margin = 0.07 * span
    ax.set_xlim(y_min - margin, y_max + margin)
    ax.set_ylim(z_min - margin, z_max + margin)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.55)
    ax.axvline(0, color="#555555", linewidth=0.8, alpha=0.55)
    ax.grid(True, linewidth=0.65, alpha=0.32)
    ax.set_xlabel(r"$y'$ — координата $v$")
    ax.set_ylabel(r"$z'$ — координата $s$")
    ax.set_title(
        "Крива перетину в системі початкової площини $x'=0$\n"
        rf"$r={parameters.r:g}$, $p={parameters.p:g}$, "
        rf"$\varphi={parameters.phi_deg:g}^\circ$, "
        rf"$\gamma={parameters.gamma_deg:g}^\circ$, "
        rf"$\lambda={parameters.lambda_deg:g}^\circ$"
    )
    ax.legend(loc="best")
    #colorbar = fig.colorbar(collection, ax=ax, pad=0.025)
    #colorbar.set_label(r"параметр $\theta$")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform the intersection to x'=0 and plot it in y'z'."
    )
    parser.add_argument("--r", type=float, default=2.536)
    parser.add_argument("--p", type=float, default=8.0)
    parser.add_argument("--phi", type=float, default=15.0, help="phi in degrees")
    parser.add_argument(
        "--gamma", type=float, default=30.0, help="gamma in degrees"
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=10.0,
        help="lambda in degrees",
    )
    parser.add_argument("--u-min", type=float, default=-2.5)
    parser.add_argument("--u-max", type=float, default=4.0)
    parser.add_argument(
        "--turns",
        type=float,
        default=2.0,
        help="theta range is [-turns*pi, turns*pi]",
    )
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("curve_initial_plane.png"),
    )
    parser.add_argument(
        "--show", action="store_true", help="also open a GUI window"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.u_min >= args.u_max:
        raise ValueError("--u-min must be smaller than --u-max")
    if args.turns <= 0 or args.samples < 2 or args.dpi <= 0:
        raise ValueError("--turns and --dpi must be positive; --samples >= 2")

    parameters = PlotParameters(
        r=args.r,
        p=args.p,
        phi_deg=args.phi,
        gamma_deg=args.gamma,
        lambda_deg=args.lam,
        u_min=args.u_min,
        u_max=args.u_max,
        turns=args.turns,
        samples=args.samples,
    )
    result = render_flattened_curve(
        parameters, args.output, dpi=args.dpi, show=args.show
    )
    print(result)


if __name__ == "__main__":
    main()
