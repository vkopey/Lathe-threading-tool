"""Symbolic intersection of a ruled screw surface and a plane.

All geometric transformations are applied to homogeneous column vectors.
The module requires SymPy 1.14.0 and does not perform any visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp
from sympy.solvers.solveset import NonlinearError


def rotation_x(angle: sp.Expr) -> sp.Matrix:
    """Return the 4 x 4 active rotation matrix about the x axis."""
    return sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle), 0],
            [0, sp.sin(angle), sp.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_y(angle: sp.Expr) -> sp.Matrix:
    """Return the 4 x 4 active rotation matrix about the y axis."""
    return sp.Matrix(
        [
            [sp.cos(angle), 0, sp.sin(angle), 0],
            [0, 1, 0, 0],
            [-sp.sin(angle), 0, sp.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_z(angle: sp.Expr) -> sp.Matrix:
    """Return the 4 x 4 active rotation matrix about the z axis."""
    return sp.Matrix(
        [
            [sp.cos(angle), -sp.sin(angle), 0, 0],
            [sp.sin(angle), sp.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def translation(dx: sp.Expr = 0, dy: sp.Expr = 0, dz: sp.Expr = 0) -> sp.Matrix:
    """Return the 4 x 4 translation matrix."""
    return sp.Matrix(
        [
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1],
        ]
    )


def xyz(vector: sp.Matrix) -> sp.Matrix:
    """Extract the Cartesian part of a homogeneous column vector."""
    return vector[:3, 0]


@dataclass(frozen=True)
class Geometry:
    """All symbolic objects used in the problem."""

    u: sp.Symbol
    theta: sp.Symbol
    v: sp.Symbol
    s: sp.Symbol
    r: sp.Symbol
    p: sp.Symbol
    phi: sp.Symbol
    gamma: sp.Symbol
    lam: sp.Symbol
    h: sp.Expr
    generator: sp.Matrix
    screw_surface: sp.Matrix
    plane_surface: sp.Matrix
    plane_normal: sp.Matrix
    plane_equation: sp.Expr


def build_geometry() -> Geometry:
    """Construct the generator, screw surface, and plane by matrix products."""
    u, theta, v, s = sp.symbols("u theta v s", real=True)
    r, p = sp.symbols("r p", real=True)
    phi, gamma, lam = sp.symbols("phi gamma lambda", real=True)
    h = p / (2 * sp.pi)

    initial_line = sp.Matrix([0, u, 0, 1])
    generator = sp.simplify(translation(dy=r) * rotation_x(phi) * initial_line)

    screw_motion = translation(dz=h * theta) * rotation_z(theta)
    screw_surface = sp.simplify(screw_motion * generator)

    initial_plane = sp.Matrix([0, v, s, 1])
    plane_surface = sp.simplify(
        translation(dy=r) * rotation_y(lam) * rotation_z(gamma) * initial_plane
    )

    plane_xyz = xyz(plane_surface)
    plane_point = sp.simplify(plane_xyz.subs({v: 0, s: 0}))
    # The order selects the negative of the transformed (1, 0, 0) normal.
    plane_normal = sp.simplify(plane_xyz.diff(s).cross(plane_xyz.diff(v)))
    x, y, z = sp.symbols("x y z", real=True)
    plane_equation = sp.expand(plane_normal.dot(sp.Matrix([x, y, z]) - plane_point))

    return Geometry(
        u=u,
        theta=theta,
        v=v,
        s=s,
        r=r,
        p=p,
        phi=phi,
        gamma=gamma,
        lam=lam,
        h=h,
        generator=generator,
        screw_surface=screw_surface,
        plane_surface=plane_surface,
        plane_normal=plane_normal,
        plane_equation=plane_equation,
    )


def intersect_parametric_surfaces(
    surface_1: sp.Matrix,
    parameters_1: Sequence[sp.Symbol],
    surface_2: sp.Matrix,
    parameters_2: Sequence[sp.Symbol],
    free_parameter: sp.Symbol,
) -> list[dict[sp.Symbol, sp.Expr]]:
    """Solve S1(parameters_1) = S2(parameters_2).

    Two parametric surfaces give three coordinate equations in four parameters.
    One chosen parameter is kept free and the other three are solved for.  The
    exact linear solver is used when possible, with ``sympy.solve`` as a
    fallback for nonlinear systems.  Each returned dictionary describes one
    intersection branch.
    """
    all_parameters = tuple(parameters_1) + tuple(parameters_2)
    if free_parameter not in all_parameters:
        raise ValueError("free_parameter must be one of the surface parameters")

    unknowns = tuple(parameter for parameter in all_parameters if parameter != free_parameter)
    if len(unknowns) != 3:
        raise ValueError("exactly four distinct surface parameters are required")

    equations = [
        sp.expand(surface_1[i, 0] - surface_2[i, 0])
        for i in range(3)
    ]

    try:
        matrix_a, vector_b = sp.linear_eq_to_matrix(equations, unknowns)
        solution_set = sp.linsolve((matrix_a, vector_b), unknowns)
        if solution_set is sp.EmptySet:
            return []
        return [dict(zip(unknowns, solution)) for solution in solution_set]
    except NonlinearError:
        return sp.solve(equations, unknowns, dict=True, simplify=False)


def intersect_parametric_surface_and_plane(
    surface: sp.Matrix,
    surface_parameters: Sequence[sp.Symbol],
    plane: sp.Matrix,
    plane_parameters: Sequence[sp.Symbol],
    free_parameter: sp.Symbol,
) -> list[dict[sp.Symbol, sp.Expr]]:
    """Intersect a parametric surface with a parametric affine plane.

    The two plane parameters are eliminated geometrically: the cross product
    of the plane tangent vectors provides an implicit plane equation.  This is
    equivalent to solving all three equations ``surface == plane`` but is much
    faster for fully symbolic angles.
    """
    if len(surface_parameters) != 2 or len(plane_parameters) != 2:
        raise ValueError("the surface and the plane must each have two parameters")
    if free_parameter not in surface_parameters:
        raise ValueError("free_parameter must be a surface parameter")

    solved_parameters = [
        parameter for parameter in surface_parameters if parameter != free_parameter
    ]
    if len(solved_parameters) != 1:
        raise ValueError("surface parameters must be distinct")
    solved_parameter = solved_parameters[0]

    plane_xyz = xyz(plane)
    plane_point = plane_xyz.subs(dict.fromkeys(plane_parameters, 0))
    tangent_1 = plane_xyz.diff(plane_parameters[0])
    tangent_2 = plane_xyz.diff(plane_parameters[1])
    normal = tangent_2.cross(tangent_1).applyfunc(sp.trigsimp)
    equation = sp.expand(normal.dot(xyz(surface) - plane_point))

    coefficient = equation.coeff(solved_parameter)
    constant = equation.subs(solved_parameter, 0)
    if coefficient != 0 and sp.expand(equation - coefficient * solved_parameter - constant) == 0:
        return [{solved_parameter: -constant / coefficient}]

    return sp.solve(equation, solved_parameter, dict=True, simplify=False)


@dataclass(frozen=True)
class Intersection:
    """Regular branch and the scalar equation governing singular rulings."""

    A: sp.Expr
    D: sp.Expr
    C: sp.Expr
    u_of_theta: sp.Expr
    rho_of_theta: sp.Expr
    curve: sp.Matrix


def derive_intersection(geometry: Geometry) -> Intersection:
    """Derive the generic one-parameter intersection curve."""
    g = geometry
    branches = intersect_parametric_surface_and_plane(
        g.screw_surface,
        (g.u, g.theta),
        g.plane_surface,
        (g.v, g.s),
        free_parameter=g.theta,
    )
    if len(branches) != 1:
        raise RuntimeError(f"expected one generic branch, obtained {len(branches)}")

    A = (
        sp.cos(g.gamma) * sp.cos(g.lam) * sp.sin(g.theta)
        - sp.sin(g.gamma) * sp.cos(g.theta)
    )
    D = (
        sp.cos(g.phi) * A
        + sp.cos(g.gamma) * sp.sin(g.lam) * sp.sin(g.phi)
    )
    C = (
        g.r * (A + sp.sin(g.gamma))
        + sp.cos(g.gamma) * sp.sin(g.lam) * g.h * g.theta
    )
    u_of_theta = sp.factor(-C / D)
    rho_of_theta = sp.factor(g.r + sp.cos(g.phi) * u_of_theta)

    # Confirm that the compact notation is exactly the result of the general
    # surface--plane routine; only expansion is needed, not costly elimination.
    solved_u = branches[0][g.u]
    solved_numerator, solved_denominator = sp.fraction(solved_u)
    assert sp.expand(solved_numerator * D + C * solved_denominator) == 0

    curve = sp.Matrix(
        [
            -rho_of_theta * sp.sin(g.theta),
            rho_of_theta * sp.cos(g.theta),
            u_of_theta * sp.sin(g.phi) + g.h * g.theta,
        ]
    )

    return Intersection(
        A=A,
        D=D,
        C=C,
        u_of_theta=u_of_theta,
        rho_of_theta=rho_of_theta,
        curve=curve,
    )


def verify(geometry: Geometry, intersection: Intersection) -> None:
    """Run exact symbolic consistency checks; raise AssertionError on failure."""
    g = geometry
    i = intersection
    x, y, z = sp.symbols("x y z", real=True)

    # Every point of the constructed plane satisfies its implicit equation.
    plane_check = g.plane_equation.subs(
        dict(zip((x, y, z), xyz(g.plane_surface)))
    )
    assert sp.trigsimp(plane_check) == 0

    # Substitution of the screw surface gives exactly D(theta)*u + C(theta).
    surface_in_plane = g.plane_equation.subs(
        dict(zip((x, y, z), xyz(g.screw_surface)))
    )
    assert sp.trigsimp(sp.expand_trig(surface_in_plane) - (i.D * g.u + i.C)) == 0

    # The regular curve lies on both input surfaces.
    screw_point = xyz(g.screw_surface).subs(g.u, i.u_of_theta)
    assert all(sp.cancel(a - b) == 0 for a, b in zip(screw_point, i.curve))
    assert sp.cancel((i.D * g.u + i.C).subs(g.u, i.u_of_theta)) == 0
    curve_in_plane = g.plane_equation.subs(
        dict(zip((x, y, z), i.curve))
    )
    assert sp.trigsimp(sp.cancel(curve_in_plane)) == 0


def main() -> None:
    geometry = build_geometry()
    intersection = derive_intersection(geometry)
    verify(geometry, intersection)

    print(f"SymPy: {sp.__version__}")
    print("\nGenerator L(u) =")
    sp.pprint(geometry.generator)
    print("\nScrew surface S(u, theta) =")
    sp.pprint(geometry.screw_surface)
    print("\nPlane P(v, s) =")
    sp.pprint(geometry.plane_surface)
    print("\nPlane normal n =")
    sp.pprint(geometry.plane_normal)
    print("\nPlane equation Pi(x, y, z) = 0, where Pi =")
    sp.pprint(geometry.plane_equation)
    print("\nIntersection equation: D(theta)*u + C(theta) = 0")
    print("D(theta) =")
    sp.pprint(intersection.D)
    print("C(theta) =")
    sp.pprint(intersection.C)
    print("\nu(theta) = -C(theta)/D(theta) =")
    sp.pprint(intersection.u_of_theta)
    print("\nIntersection curve c(theta) =")
    sp.pprint(intersection.curve)
    print("\nAll symbolic checks passed.")


if __name__ == "__main__":
    main()
