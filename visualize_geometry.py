"""Matplotlib visualization of the symbolic geometry and its intersection.

The plotted coordinates are produced by ``lambdify`` directly from the SymPy
objects returned by ``build_geometry`` and ``derive_intersection``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from helicoid_plane_intersection import build_geometry, derive_intersection, xyz


@dataclass(frozen=True)
class PlotParameters:
    """Numeric values and sampling ranges used by the visualization."""

    r: float = 2.536
    p: float = 8.0
    phi_deg: float = 15.0
    gamma_deg: float = -30.0
    lambda_deg: float = 10.0
    u_min: float = -2.5
    u_max: float = 4.0
    turns: float = 2.0
    plane_span: float = 6.0
    surface_u_samples: int = 80
    surface_theta_samples: int = 180
    curve_samples: int = 1200


def _matrix_function(expression: sp.Matrix, arguments: tuple[sp.Symbol, ...]):
    """Create a NumPy function returning the three Cartesian coordinates."""
    component_functions = [
        sp.lambdify(arguments, xyz(expression)[index], modules="numpy", cse=True)
        for index in range(3)
    ]

    def evaluate(*values: object) -> np.ndarray:
        target_shape = np.broadcast_arrays(*values)[0].shape
        components = [
            np.broadcast_to(
                np.asarray(function(*values), dtype=float), target_shape
            )
            for function in component_functions
        ]
        return np.stack(components, axis=0)

    return evaluate


def _coordinates(values: object) -> np.ndarray:
    """Normalize lambdified Matrix output to an array whose first axis is xyz."""
    result = np.asarray(values, dtype=float)
    return np.squeeze(result)


def _set_equal_3d_limits(ax, coordinate_sets: list[np.ndarray]) -> None:
    """Set equal data scale on all three axes."""
    finite_sets = []
    for coordinates in coordinate_sets:
        flattened = coordinates.reshape(3, -1)
        finite_columns = np.all(np.isfinite(flattened), axis=0)
        if np.any(finite_columns):
            finite_sets.append(flattened[:, finite_columns])

    all_points = np.concatenate(finite_sets, axis=1)
    minima = np.min(all_points, axis=1)
    maxima = np.max(all_points, axis=1)
    centers = (minima + maxima) / 2
    radius = max(np.max(maxima - minima) / 2, 1.0) * 1.04

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_geometry(
    parameters: PlotParameters,
    output_path: Path,
    *,
    dpi: int = 180,
    show: bool = False,
) -> Path:
    """Render every geometric object and save the resulting figure."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    geometry = build_geometry()
    intersection = derive_intersection(geometry)

    phi = np.deg2rad(parameters.phi_deg)
    gamma = np.deg2rad(parameters.gamma_deg)
    lam = np.deg2rad(parameters.lambda_deg)
    substitutions = {
        geometry.r: parameters.r,
        geometry.p: parameters.p,
        geometry.phi: phi,
        geometry.gamma: gamma,
        geometry.lam: lam,
    }

    theta_limit = parameters.turns * np.pi
    theta_values = np.linspace(
        -theta_limit, theta_limit, parameters.surface_theta_samples
    )
    u_values = np.linspace(
        parameters.u_min, parameters.u_max, parameters.surface_u_samples
    )
    theta_mesh, u_mesh = np.meshgrid(theta_values, u_values)

    screw_function = _matrix_function(
        geometry.screw_surface.subs(substitutions), (geometry.u, geometry.theta)
    )
    screw_coordinates = _coordinates(screw_function(u_mesh, theta_mesh))

    generator_function = _matrix_function(
        geometry.generator.subs(substitutions), (geometry.u,)
    )
    generator_coordinates = _coordinates(generator_function(u_values))

    plane_values = np.linspace(
        -parameters.plane_span, parameters.plane_span, 35
    )
    v_mesh, s_mesh = np.meshgrid(plane_values, plane_values)
    plane_function = _matrix_function(
        geometry.plane_surface.subs(substitutions), (geometry.v, geometry.s)
    )
    plane_coordinates = _coordinates(plane_function(v_mesh, s_mesh))

    curve_theta = np.linspace(-theta_limit, theta_limit, parameters.curve_samples)
    curve_function = _matrix_function(
        intersection.curve.subs(substitutions), (geometry.theta,)
    )
    curve_coordinates = _coordinates(curve_function(curve_theta))

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
    curve_u = np.asarray(u_function(curve_theta), dtype=float)
    denominator = np.asarray(denominator_function(curve_theta), dtype=float)
    denominator_scale = max(float(np.nanmax(np.abs(denominator))), 1.0)
    regular = np.abs(denominator) > 1e-7 * denominator_scale
    on_surface_patch = (
        (curve_u >= parameters.u_min) & (curve_u <= parameters.u_max)
    )
    valid_curve = regular & on_surface_patch & np.all(
        np.isfinite(curve_coordinates), axis=0
    )
    curve_coordinates[:, ~valid_curve] = np.nan

    normal = _coordinates(geometry.plane_normal.subs(substitutions)).reshape(3)
    normal /= np.linalg.norm(normal)
    plane_point = _coordinates(
        xyz(
            geometry.plane_surface.subs(substitutions).subs(
                {geometry.v: 0, geometry.s: 0}
            )
        )
    ).reshape(3)
    normal_length = 0.32 * parameters.plane_span

    fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        screw_coordinates[0],
        screw_coordinates[1],
        screw_coordinates[2],
        color="#3b82b8",
        alpha=0.38,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.plot_wireframe(
        screw_coordinates[0],
        screw_coordinates[1],
        screw_coordinates[2],
        rstride=8,
        cstride=15,
        color="#276388",
        linewidth=0.35,
        alpha=0.30,
    )
    ax.plot_surface(
        plane_coordinates[0],
        plane_coordinates[1],
        plane_coordinates[2],
        color="#e0a32f",
        alpha=0.24,
        linewidth=0,
        shade=False,
    )
    ax.plot_wireframe(
        plane_coordinates[0],
        plane_coordinates[1],
        plane_coordinates[2],
        rstride=5,
        cstride=5,
        color="#a66f10",
        linewidth=0.4,
        alpha=0.22,
    )
    ax.plot(
        generator_coordinates[0],
        generator_coordinates[1],
        generator_coordinates[2],
        color="#163d64",
        linewidth=3.0,
        zorder=7,
    )
    ax.plot(
        curve_coordinates[0],
        curve_coordinates[1],
        curve_coordinates[2],
        color="#c62828",
        linewidth=3.2,
        zorder=10,
    )
    ax.quiver(
        plane_point[0],
        plane_point[1],
        plane_point[2],
        normal[0],
        normal[1],
        normal[2],
        length=normal_length,
        normalize=True,
        color="#27824b",
        linewidth=2.4,
        arrow_length_ratio=0.18,
    )

    z_axis_extent = max(
        float(np.nanmax(np.abs(screw_coordinates[2]))),
        float(np.nanmax(np.abs(curve_coordinates[2]))),
        1.0,
    )
    ax.plot(
        [0, 0],
        [0, 0],
        [-z_axis_extent, z_axis_extent],
        color="#4d4d4d",
        linewidth=1.2,
        linestyle="--",
        alpha=0.75,
    )

    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=8)
    ax.set_title(
        "Перетин гвинтової поверхні з площиною\n"
        rf"$r={parameters.r:g}$, $p={parameters.p:g}$, "
        rf"$\varphi={parameters.phi_deg:g}^\circ$, "
        rf"$\gamma={parameters.gamma_deg:g}^\circ$, "
        rf"$\lambda={parameters.lambda_deg:g}^\circ$",
        pad=18,
    )
    ax.view_init(elev=24, azim=-56)
    ax.grid(True, alpha=0.30)

    legend_handles = [
        Patch(facecolor="#3b82b8", alpha=0.55, label="Helix"),
        Patch(facecolor="#e0a32f", alpha=0.40, label="Plane"),
        Line2D([0], [0], color="#163d64", linewidth=3, label="Generatrix L(u)"),
        Line2D(
            [0], [0], color="#c62828", linewidth=3, label="Curve"
        ),
        Line2D(
            [0],
            [0],
            color="#27824b",
            linewidth=2.4,
            marker=">",
            label="Plane normal",
        ),
        Line2D(
            [0],
            [0],
            color="#4d4d4d",
            linewidth=1.2,
            linestyle="--",
            label="Helix axis z",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.92)

    limit_coordinates = [
        screw_coordinates,
        plane_coordinates,
        generator_coordinates,
        curve_coordinates,
    ]
    _set_equal_3d_limits(ax, limit_coordinates)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the screw surface, plane, and intersection curve."
    )
    parser.add_argument("--r", type=float, default=2.536)
    parser.add_argument("--p", type=float, default=8.0)
    parser.add_argument("--phi", type=float, default=15.0, help="phi in degrees")
    parser.add_argument("--gamma", type=float, default=-30.0, help="gamma in degrees")
    parser.add_argument("--lambda", dest="lam", type=float, default=10.0, help="lambda in degrees")
    parser.add_argument("--u-min", type=float, default=-2.5)
    parser.add_argument("--u-max", type=float, default=4.0)
    parser.add_argument("--turns", type=float, default=2.0, help="theta range is [-turns*pi, turns*pi]")
    parser.add_argument("--plane-span", type=float, default=6.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output", type=Path, default=Path("geometry_intersection.png")
    )
    parser.add_argument("--show", action="store_true", help="also open a GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.u_min >= args.u_max:
        raise ValueError("--u-min must be smaller than --u-max")
    if args.turns <= 0 or args.plane_span <= 0 or args.dpi <= 0:
        raise ValueError("--turns, --plane-span, and --dpi must be positive")

    parameters = PlotParameters(
        r=args.r,
        p=args.p,
        phi_deg=args.phi,
        gamma_deg=args.gamma,
        lambda_deg=args.lam,
        u_min=args.u_min,
        u_max=args.u_max,
        turns=args.turns,
        plane_span=args.plane_span,
    )
    result = render_geometry(
        parameters, args.output, dpi=args.dpi, show=args.show
    )
    print(result)


if __name__ == "__main__":
    main()
