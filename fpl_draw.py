import fastplotlib as fpl
import numpy as np
import sympy as sp
import pygfx as gfx
from numpy import sin, cos, tan, pi

def drawSurf(figure,u,t,ur,tr,X,Y,Z,args={}):
    """Рисує поверхню, задану SymPy виразами X,Y,Z з параметрами u,t
    figure - фігура
    u,t - SymPy-параметри
    ur,tr - числові межі параметрів
    X,Y,Z - SymPy-вирази
    args - словник з константами для підстановки в SymPy-вирази
    Приклад:
    drawSurf(figure, t, u, [0, 1], [0, 1], X, Y, Z, {R:2})
"""
    u_=np.linspace(*ur)
    t_=np.linspace(*tr)
    u_, t_ = np.meshgrid(u_, t_)
    xyz=[]
    for i_ in X,Y,Z:
        i=i_.subs(args)
        if i.is_number: # якщо число
            xyz.append(0*u_+i.evalf()) # заповнити масив цим числом
            print("warning! is_number")
        else:
            f=sp.lambdify([u,t], i, "numpy")
            xyz.append(f(u_,t_))
    surface = figure[0, 0].add_surface(np.dstack(xyz), cmap="bwr", alpha=0.2)
    figure[0, 0].camera.show_object(surface.world_object, (-2, 2, -3), up=(0, 0, 1))


def drawCurve(figure, u, ur, X, Y, Z, args={}, colors="green"):
    u_=np.linspace(*ur)
    xyz=[]
    for i_ in X,Y,Z:
        i=i_.subs(args)
        if i.is_number: # якщо число
            xyz.append(0*u_+i.evalf()) # заповнити масив цим числом
            print("warning! is_number")
        else:
            f=sp.lambdify(u, i, "numpy")
            xyz.append(f(u_))
    figure[0, 0].add_line(data=np.column_stack(xyz), thickness=8, colors=colors)

figure = fpl.Figure(size=(700, 560), cameras="3d", controller_types="orbit")


if __name__ == "__main__":
    #figure[0, 0].axes.grids.xy.visible = True
    drawSurf()
    drawCurve()
    figure.show()
    fpl.loop.run()