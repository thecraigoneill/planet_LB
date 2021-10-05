import numpy as np
#from numba import jit

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
      'x':x,
    }

    #T = self.LB1D(LB_struct)

    #return(T,x)
    return(LB_struct)

  #@jit(nopython=True)
  def loopy_loop(self,n,m,vx,vy,cx,cy,rho,w):
    feq = np.zeros((9,n+1,m+1))
    for i in range(0,n+1):
        for j in range(0,m+1):
            t1 = vx[i,j]*vx[i,j] + vy[i,j]*vy[i,j]
            for k in range(0,9):
                t2 = vx[i,j]*cx[k] + vy[i,j]*cy[k]
                feq[k,i,j] = rho[i,j]*w[k]* (1.0+3.0*t2+4.50*t2*t2-1.50*t1)
    return feq

  #@jit(nopython=True)
  def LB_D2Q9_V(self,LB_struct_2D):
    n = LB_struct_2D['n']
    m = LB_struct_2D['m']
    vx = LB_struct_2D['vx']
    vy = LB_struct_2D['vy']
    cx = LB_struct_2D['cx']
    cy = LB_struct_2D['cy']
    rho = LB_struct_2D['rho']
    dt = LB_struct_2D['dt']
    w = LB_struct_2D['w']
    H = LB_struct_2D['H']
    uo = LB_struct_2D['uo']
    omegaV = LB_struct_2D['omega']
    f = LB_struct_2D['f']



    feq = self.loopy_loop(n,m,vx,vy,cx,cy,rho,w)
    #Collision step
    f[0,:,:] = (1.0 - omegaV)*f[0,:,:] + omegaV*feq[0,:,:]  + dt*0.5*H
    f[1,:,:] = (1.0 - omegaV)*f[1,:,:] + omegaV*feq[1,:,:]  + dt*0.5*H
    f[2,:,:] = (1.0 - omegaV)*f[2,:,:] + omegaV*feq[2,:,:]  + dt*0.5*H
    f[3,:,:] = (1.0 - omegaV)*f[3,:,:] + omegaV*feq[3,:,:]  + dt*0.5*H
    f[4,:,:] = (1.0 - omegaV)*f[4,:,:] + omegaV*feq[4,:,:]  + dt*0.5*H
    f[5,:,:] = (1.0 - omegaV)*f[5,:,:] + omegaV*feq[5,:,:]  + dt*0.5*H
    f[6,:,:] = (1.0 - omegaV)*f[6,:,:] + omegaV*feq[6,:,:]  + dt*0.5*H
    f[7,:,:] = (1.0 - omegaV)*f[7,:,:] + omegaV*feq[7,:,:]  + dt*0.5*H
    f[8,:,:] = (1.0 - omegaV)*f[8,:,:] + omegaV*feq[8,:,:]  + dt*0.5*H

    #Streaming step  # 

    f[1,:,:] = np.roll(f[1,:,:],1,axis=0)
    f[2,:,:] = np.roll(f[2,:,:],1,axis=1)
    f[3,:,:] = np.roll(f[3,:,:],-1,axis=0)
    f[4,:,:] = np.roll(f[4,:,:],-1,axis=1)

    f[5,:,:] = np.roll(f[5,:,:],1,axis=0)
    f[5,:,:] = np.roll(f[5,:,:],1,axis=1)
    f[6,:,:] = np.roll(f[6,:,:],-1,axis=0)
    f[6,:,:] = np.roll(f[6,:,:],1,axis=1)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=0)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=1)    
    f[8,:,:] = np.roll(f[8,:,:],1,axis=0)
    f[8,:,:] = np.roll(f[8,:,:],-1,axis=1)
  

    # Bounce back
    # West
    f[1,0,:]=f[3,0,:]
    f[5,0,:]=f[7,0,:]
    f[8,0,:]=f[6,0,:]
    #East
    f[3,n,:]=f[1,n,:]
    f[7,n,:]=f[5,n,:]
    f[6,n,:]=f[8,n,:]
    # South
    f[2,:,0]=f[4,:,0]
    f[5,:,0]=f[7,:,0]
    f[6,:,0]=f[8,:,0]
    #North - moving lid
    f[4,:,m] = f[2,:,m]
    rhon = f[0,:,m] + f[1,:,m] + f[3,:,m] + 2.0*(f[2,:,m] + f[6,:,m] + f[5,:,m])
    f[8,:,m] = f[6,:,m] + rhon*uo/6.0
    f[7,:,m] = f[5,:,m] - rhon*uo/6.0

    rho = f[0,:,:] + f[1,:,:] + f[2,:,:] + f[3,:,:] + f[4,:,:] +f[5,:,:] + f[6,:,:]+f[7,:,:]+f[8,:,:]
    rho[:,m] = f[0,:,m] + f[1,:,m] + f[3,:,m] + 2.0*(f[2,:,m] + f[6,:,m] + f[5,:,m])

    usum = f[0,:,:]*cx[0] + f[1,:,:]*cx[1] + f[2,:,:]*cx[2] + f[3,:,:]*cx[3] + f[4,:,:]*cx[4] +f[5,:,:]*cx[5] + f[6,:,:]*cx[6] +f[7,:,:]*cx[7]+f[8,:,:]*cx[8]
    vsum = f[0,:,:]*cy[0] + f[1,:,:]*cy[1] + f[2,:,:]*cy[2] + f[3,:,:]*cy[3] + f[4,:,:]*cy[4] +f[5,:,:]*cy[5] + f[6,:,:]*cy[6] +f[7,:,:]*cy[7]+f[8,:,:]*cy[8]
    vx = usum/rho
    vy = vsum/rho

    return rho, vx, vy

  #dt,m,n,cx,cy,omegaV,w,rho,vx,vy,f,feq,uo,H
  def LB_D2Q9_init_Lid(self,dt,m,n,dx,dy,rhoo,uo,H,alpha,mstep):
            
    # Initiatialise grid
    x = np.arange(0,dx*(n+1),dx)
    y = np.arange(0,dy*(m+1),dy)
    #print(y)
    X,Y = np.meshgrid(x,y) # Change rows to columns
    X=X.T
    Y=Y.T

    f=np.zeros((9,n+1,m+1))
    feq=np.zeros((9,n+1,m+1))

    #global rho
    rho=np.ones((n+1,m+1))*rhoo
    #global vx
    vx=np.zeros((n+1,m+1))
    #global vy
    vy=np.zeros((n+1,m+1))

    vx[1:n-1,m]=uo
    vy[1:n-1,m]=0.


    w=np.zeros(9)
    w[0]=4./9.
    w[1]=w[2]=w[3]=w[4]=1./9.
    w[5]=w[6]=w[7]=w[8]=1./36.

    cx = np.array([0.0,1.0,0.0,-1.0,0.0,1.0,-1.0,-1.0,1.0])
    cy = np.array([0.0,0.0,1.0,0.0,-1.0,1.0,1.0,-1.0,-1.0])

    f[0,:,:] =  w[0]*rho
    f[1,:,:] =  w[1]*rho
    f[2,:,:] =  w[2]*rho
    f[3,:,:] =  w[3]*rho
    f[4,:,:] =  w[4]*rho
    f[5,:,:] =  w[5]*rho
    f[6,:,:] =  w[6]*rho
    f[7,:,:] =  w[7]*rho
    f[8,:,:] =  w[8]*rho
    

    omegaV=1.0/(3.*alpha+0.5) 
    #T = np.ones((m+1))*Tinit
    #   self,dt,dx,mstep,m,Tinit,TSurf,H

    LB_struct = {
      'dt': dt,
      'mstep': mstep,
      'm': m,
      'n':n,
      'vx':vx,
      'vy':vy,
      'cx':cx,
      'cy':cy,
      'uo': uo,
      'rho': rho,
      'H': H,
      'w':w,
      'omega': omegaV,
      'f':f,
      'feq':feq,
      'x':X,
      'y':Y,
    }
    return LB_struct

