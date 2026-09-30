import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import ezdxf

pi,sin,cos,tan=np.pi,np.sin,np.cos,np.tan
r = 2.536
p = 8.0
Xstart=7.5
Xstop=12.0
Ystart=2.119, -1.737
Ystop=3.705, -2.586
phi = np.radians(15.0)
gamma = np.radians(-30.0) #-30 -10
lam = np.radians(10.0) #10

theta=np.linspace(0.3, -0.47)
#theta=np.linspace(.0, -.183)
#theta=np.linspace(.0, -.11)

def xyz():
    h=p/(2*pi)
    A = cos(gamma) * cos(lam) * sin(theta) - sin(gamma) * cos(theta)
    D = cos(phi) * A + cos(gamma) * sin(lam) * sin(phi)
    C = r * (A + sin(gamma)) + cos(gamma) * sin(lam) * h * theta
    u_of_theta = -C / D
    rho_of_theta = r + cos(phi) * u_of_theta
    x=-rho_of_theta * sin(theta)
    y=rho_of_theta * cos(theta)
    z=u_of_theta * sin(phi) + h * theta
    return y,z

def yz():
    E=(2*pi*(-sin(gamma)*cos(phi)*cos(theta) + sin(lam)*sin(phi)*cos(gamma) + sin(theta)*cos(gamma)*cos(lam)*cos(phi)))
    y=(-p*theta*sin(lam)*cos(phi)*cos(theta) + 2*pi*r*sin(lam)*sin(phi)*cos(theta) - 2*pi*r*sin(lam)*sin(phi) - 2*pi*r*sin(theta)*cos(lam)*cos(phi))/E
    z=(-p*theta*sin(gamma)*cos(lam)*cos(phi)*cos(theta) + p*theta*sin(theta)*cos(gamma)*cos(phi) + 2*pi*r*sin(gamma)*sin(lam)*sin(theta)*cos(phi) + 2*pi*r*sin(gamma)*sin(phi)*cos(lam)*cos(theta) - 2*pi*r*sin(gamma)*sin(phi)*cos(lam) - 2*pi*r*sin(phi)*sin(theta)*cos(gamma))/E
    return y,z

def readDXF(filename, n): # n - номер сплайну
  doc = ezdxf.readfile(filename)
  msp = doc.modelspace()
  X=[]
  Y=[]
  splines_data = []
  # Шукаємо об'єкти SPLINE
  for i, spline in enumerate(msp.query("SPLINE")):
    print(f"\nЗнайдено сплайн #{i+1}:")

    # Варіант А: Отримати керуючі/опорні точки (Control Points)
    control_pts = [(p[0], p[1], p[2]) for p in spline.control_points]
    print(f" - Контрольних точок: {len(control_pts)}")

    # Варіант Б: Дискретизувати сплайн на N точок уздовж кривої (найкраще для порівняння!)
    # Створює N точок з високою точністю вздовж самої кривої
    flattened_pts = list(spline.flattening(distance=0.01))  # distance = точність/крок
    print(f" - Згенеровано точок вздовж кривої: {len(flattened_pts)}")
    splines_data.append(flattened_pts)
    # Тепер у splines_data[0] та splines_data[1] лежать масиви точок (X, Y, Z) для двох кривих

  X=[p[0] for p in splines_data[n]]
  Y=[p[1] for p in splines_data[n]]
  return np.array(Y),np.array(X)

def rotate(X,Y,a):
    return X*cos(a)-Y*sin(a), X*sin(a)+Y*cos(a)

def R2(X1,Y1,X2,Y2):
    """Обчислює R^2 двох масивів точок, які мають різні ординати len(X1)>len(X2)"""
    f = interp1d(X1, Y1, kind='linear')
    return r2_score(Y2, f(X2)), np.sqrt(np.mean((Y2-f(X2))**2)), np.max(np.fabs(Y2-f(X2)))

def interp(X,Y,Xstart,Xstop):
    """Інтерполяція кривої X,Y з межами Xstart,Xstop"""
    f = interp1d(X, Y, kind='linear')
    X=np.linspace(Xstart,Xstop)
    return X, f(X)

def lin_approx(X,Y):
    """Показники якості лінійної апроксимації даних"""
    f=lambda x,a,b: a*x+b
    popt, pcov = curve_fit(f, X, Y)
    return r2_score(Y, f(X,*popt)), np.sqrt(np.mean((Y-f(X,*popt))**2)), np.max(np.fabs(Y-f(X,*popt)))

Y1,X1=readDXF("RakeAngle2_.DXF",0) # отримати криву з CAD
Y2,X2=readDXF("RakeAngle2_.DXF",1)
Y2,X2=Y2[::-1],X2[::-1] # обернути масиви

# зміщення в нуль
X1=X1-X1[0]
Y1=Y1-Y1[0]
X2=X2-X2[0]
Y2=Y2-Y2[0]
X1,Y1=-X1,Y1 # відобразити дзеркально відносно осі Y
X2,Y2=-X2,Y2

# обертання
a=0.719
X1,Y1=rotate(X1,Y1,a)
X2,Y2=rotate(X2,Y2,a)

plt.plot(X1,Y1,'o-') # коригований (CAD, правий)
plt.plot(X2,Y2,'^-') # коригований (CAD, лівий)

plt.plot(*xyz(),'r--')
plt.plot(*yz(),'r') # коригований (SymPy, правий)
plt.plot([0,0+5],[0, 5*tan(phi)],'k:') # стандартний правий
print("R2=",R2(*yz(),X1,Y1)) # R2(SymPy, CAD) правий
#print("R2=",R2(yz()[1],yz()[0],Y1,X1))
print(lin_approx(*interp(*yz(),Xstart,Xstop))) # якість лінійної апроксимації (правий)
#print(lin_approx(*interp(yz()[1],yz()[0],Ystart[0],Ystop[0])))

phi = np.radians(-15.0)
plt.plot(*xyz(),'b--')
plt.plot(*yz(),'b') # коригований (SymPy, лівий)
plt.plot([0,0+5],[0, 5*tan(phi)],'k:') # стандартний лівий
print("R2=",R2(*yz(),X2,Y2)) # R2(SymPy, CAD) лівий
#print("R2=",R2(yz()[1],yz()[0],Y2,X2))
print(lin_approx(*interp(*yz(),Xstart,Xstop))) # якість лінійної апроксимації (лівий)
#print(lin_approx(*interp(yz()[1],yz()[0],Ystart[1],Ystop[1])))

plt.axis('equal')
plt.grid()
plt.xlabel('$y$, mm'); plt.ylabel("$z$, mm")
plt.show()