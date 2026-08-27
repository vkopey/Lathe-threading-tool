import numpy as np
import matplotlib.pyplot as plt

pi,sin,cos,tan=np.pi,np.sin,np.cos,np.tan
r = 2.536
p = 8.0
phi = np.radians(15.0)
gamma = np.radians(-30.0) #30
lam = np.radians(10.0) #10

theta=np.linspace(0.3, -0.3)

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

plt.plot(*xyz(),'r--')
plt.plot(*yz(),'r')
plt.plot([0,0+5],[0, 5*tan(phi)],'k:')

phi = np.radians(-15.0)
plt.plot(*xyz(),'b--')
plt.plot(*yz(),'b')
plt.plot([0,0+5],[0, 5*tan(phi)],'k:')

plt.axis('equal')
plt.grid()
plt.xlabel('$y$, mm'); plt.ylabel("$z$, mm")
plt.show()