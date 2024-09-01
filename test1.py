import numpy as np
import matplotlib.pyplot as plt
def intB(x):
    #return np.where(x%1.<0.5,np.power(np.sin(2*x*np.pi),2),0)
    return np.where(x<1.,np.sin(2*x*np.pi),0)
    
    
nx=40
c=0.2
x=np.linspace(0.0,1.0,nx+1)
phi=intB(x)
phiN=phi.copy()
phiO=phi.copy()
#FTCS FOR FIRST TIME STEP
#loop over space
for j in range(1,nx):
    phi[j]=phiO[j]-0.5*c*(phiO[j+1]-phiO[j-1])
phi[0]=phiO[0]-0.5*c*(phiO[1]-phiO[nx-1])
phi[nx]=phi[0]


#Remaining time steps
nt=40
for n in range(1,nt):
    for j in range(1,nx):
         phiN[j]=phiO[j]-c*(phi[j+1]-phi[j-1])
         #update phi for next time step
    #periodic boundary conditiobns
    phiN[0]=phiO[0]-c*(phi[1]-phi[nx-1])
    phiN[nx]=phiN[0]

    phiO=phi.copy()
    phi=phiN.copy()
    
         
         
u=1.
dx=1./nx
dt=c*dx/u
t=nt*dt

plt.plot(x,intB(x-u*t),'k',label='analytic')
plt.plot(x,phi,'b',label='CTCS')
plt.xlabel('x')
plt.ylabel('Phi')
plt.legend(loc='best')

#exitplt.axhline(0, linestyle=':',color='red')
plt.show()