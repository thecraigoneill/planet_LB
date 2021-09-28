import numpy as np

Myr2s = (60*60*24*365.25*1e6)

class planet_LB():
  def LB1D(self,LB_struct):
    mstep = LB_struct['mstep']
    m = LB_struct['m']
    dt = LB_struct['dt']
    f1  = LB_struct['f1']
    f2 = LB_struct['f2']
    H = LB_struct['H']
    omega = LB_struct['omega']
    TSurf = LB_struct['TSurf']

    timestep = dt*np.arange(1,mstep+1,1)
    n = np.arange(1,m,1)
    for kk in timestep:
        #Collision step
        T = f1 + f2
        feq = 0.5*T
        f1 = (1.0 - omega)*f1 + omega*feq + dt*0.5*H
        f2 = (1.0 - omega)*f2 + omega*feq + dt*0.5*H    # H is heat production
        #Streaming step
        for i in n:
            f1[m-i] = f1[m-i-1]   #This steps f1 to the left
            f2[i-1] = f2[i]       # This steps f2 right
        #Boundary condition
        f1[0] = TSurf - f2[0]   #constant temperature wall at x=0
        f1[m] = f1[m-1]         # adiabatic boundary condition, x=L,
        f2[m] = f2[m-1]
    return T

  def init_1D_LB(self,dt,dx,mstep,m,Tinit,TSurf,H):
    f1 = np.array([])
    f2 = np.array([])
    feq = np.array([])
    x = np.array([])
    scale=(60*60*24*365.25*1e6)
    x = np.arange(0,dx*(m+1),dx)
    ck=dx/dt
    csq=ck*ck
    K=1e-6
    #k=3
    #flux=80e-3

    omega=1.0/(K/(dt*csq)+0.5) 
    T = np.ones((m+1))*Tinit
    f1 = 0.5*T
    f2 = 0.5*T

    LB_struct = {
      'dt': dt,
      'mstep': mstep,
      'm': m,
      'TSurf': TSurf,
      'Tinit': Tinit,
      'H': H,
      'omega': omega,
      'f1':f1,
      'f2':f2,
      'feq':feq,
    }

    T = self.LB1D(LB_struct)

    return(T,x)
