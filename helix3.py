import sympy as sp
import numpy as np
from fpl_draw import fpl, figure, drawSurf, drawCurve


def homog(R): # переводить матрицю 3×3 в однорідні координати (4×4)
    return R.row_join(sp.zeros(3,1)).col_join(sp.Matrix([[0,0,0,1]]))

# Параметри
t, s, u, v, fi, gamma, lamda = sp.symbols('t s u v fi gamma lamda', real=True)
R, p, h = sp.symbols('R p h', real=True, positive=True)

args={R:2.536, fi:np.radians(15), p:8, gamma:np.radians(30), lamda:np.radians(10)}

# Крок різьби: підйом за 2π
dz = (p/(2*sp.pi))*t

# Пряма (профіль різця) у локальній системі координат
# Наприклад, пряма вздовж осі y від -h до h
line_vec = sp.Matrix([0, u, 0, 1])  # однорідні координати
drawCurve(figure, u, [0, 6], line_vec[0], line_vec[1], line_vec[2], args, "orange")

rotx=homog(sp.rot_axis1(fi))
line_vec= rotx*line_vec # повернути пряму навколо осі x
drawCurve(figure, u, [0, 6], line_vec[0], line_vec[1], line_vec[2], args, "green")

rotz=homog(sp.rot_axis3(gamma))
line_vec= rotz*line_vec # повернути пряму навколо осі z
drawCurve(figure, u, [0, 6], line_vec[0], line_vec[1], line_vec[2], args, "red")

roty=homog(sp.rot_axis2(lamda))
line_vec= roty*line_vec # повернути пряму навколо осі y
drawCurve(figure, u, [0, 6], line_vec[0], line_vec[1], line_vec[2], args, "blue")

Trans = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, R],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
line_vec=Trans*line_vec
drawCurve(figure, u, [0, 6], line_vec[0], line_vec[1], line_vec[2], args, "magenta")

# Матриця повороту навколо осі z на кут t
Rot_z = sp.Matrix([
    [sp.cos(t), -sp.sin(t), 0, 0],
    [sp.sin(t),  sp.cos(t), 0, 0],
    [0,          0,         1, 0],
    [0,          0,         0, 1]
])

# Матриця переносу до гвинтової лінії
Trans = sp.Matrix([
    [1, 0, 0, R*sp.cos(t)],
    [0, 1, 0, R*sp.sin(t)],
    [0, 0, 1, dz],
    [0, 0, 0, 1]
])
# Матриця перенесення 4х4
Trans = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dz],
    [0, 0, 0, 1]
])

# Загальна матриця перетворення
M = Trans * Rot_z

# Глобальні координати точки профілю
global_point = M * line_vec

X, Y, Z = sp.simplify(global_point[0]), sp.simplify(global_point[1]), sp.simplify(global_point[2])
drawSurf(figure, t, u, [-1, 2*2*np.pi], [-1, 3], X, Y, Z, args)

# X=X.subs(args)
# Y=Y.subs(args)
# Z=Z.subs(args)

print(X)
print(Y)
print(Z)


##
# площина
X1=u
Y1=0
Z1=s
drawSurf(figure, s, u, [-6, 6], [-6, 6], X1, Y1*u, Z1) # 0*u - щоб був символ

"""
# Система рівнянь
eqs = [
    sp.Eq(X, X1),
    sp.Eq(Y, Y1),
    sp.Eq(Z, Z1)
]

# Розв’язок відносно u,t,v
sol = sp.solve(eqs, [u, t, v], dict=True)
print(sol)

# Параметричне рівняння кривої перетину
X_curve = X.subs(sol[1])
Y_curve = Y.subs(sol[1])
Z_curve = Z.subs(sol[1])
drawCurve(figure, s, [5, 10], X_curve, Y_curve, Z_curve, args, "yellow")
"""

##
# Інший спосіб. Загальне рівняння площини: A*x + B*y + C*z + D = 0
# Підставляємо вирази поверхні замість x, y, z
#plane_eq = Y
plane_eq =-sp.sin(gamma)*X+sp.cos(gamma)*Y # площина повернута навколо z

# 4. Знаходимо залежність u від v (розв'язуємо рівняння відносно u)
u_sol = sp.solve(plane_eq, u)[0]

print("Залежність параметра u від v на лінії перетину:")
sp.pprint(sp.Eq(u, u_sol))
print("\n" + "="*50 + "\n")

# 5. Підставляємо u(v) назад у рівняння поверхні, щоб отримати криву від одного параметра v
X_curve = X.subs(u, u_sol)
Y_curve = Y.subs(u, u_sol)
Z_curve = Z.subs(u, u_sol)  # Z не залежить від u, залишається незмінним

print("Параметричне рівняння кривої перетину (залежить тільки від v):")
print(f"X_c(v) = {sp.simplify(X_curve)}")
print(f"Y_c(v) = {sp.simplify(Y_curve)}")
print(f"Z_c(v) = {sp.simplify(Z_curve)}")

drawCurve(figure, t, [0, 2.4], X_curve, Y_curve, Z_curve, args, "yellow")

figure.show()
fpl.loop.run()