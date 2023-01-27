import numpy as np
from numba import jit
from scipy import special

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

    return rho, vx, vy, f

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


  def LB_D2Q9_init_ThermStruct(self,dt,m,n,dx,dy,T0,H,kappa,mstep):
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
    T=np.ones((n+1,m+1))*T0

    w=np.zeros(9)
    w[0]=4./9.
    w[1]=w[2]=w[3]=w[4]=1./9.
    w[5]=w[6]=w[7]=w[8]=1./36.


    f[0,:,:] =  w[0]*T
    f[1,:,:] =  w[1]*T
    f[2,:,:] =  w[2]*T
    f[3,:,:] =  w[3]*T
    f[4,:,:] =  w[4]*T
    f[5,:,:] =  w[5]*T
    f[6,:,:] =  w[6]*T
    f[7,:,:] =  w[7]*T
    f[8,:,:] =  w[8]*T
    

    ck=dx/dt
    csq=ck*ck 
    omega=1.0/(3*kappa/(dt*csq)+0.5) 
    #T = np.ones((m+1))*Tinit
    #   self,dt,dx,mstep,m,Tinit,TSurf,H

    LB_struct_2D = {
      'dt': dt,
      'dx':dx,
      'dy':dy,
      'mstep': mstep,
      'm': m,
      'n':n,
      'T': T,
      'H': H,
      'kappa':kappa,
      'w':w,
      'omega': omega,
      'f':f,
      'feq':feq,
      'x':X,
      'y':Y,
      'T_top':T0,
      'T_base':T0,
    }
    return LB_struct_2D

  def LB_D2Q9_T(self,LB_struct_2D):
    n = LB_struct_2D['n']
    m = LB_struct_2D['m']
    T = LB_struct_2D['T']
    dt = LB_struct_2D['dt']
    w = LB_struct_2D['w']
    H = LB_struct_2D['H']
    omega = LB_struct_2D['omega']
    f = LB_struct_2D['f']
    feq = LB_struct_2D['feq']
    T_top = LB_struct_2D['T_top']
    T_base = LB_struct_2D['T_base']


    for i,z in np.ndenumerate(w):
        feq[i,:,:] = w[i]*T
    f = (1.0 - omega)*f + omega*feq + dt*0.5*H
    #Streaming step  # 
    f[2,:,:] = np.roll(f[2,:,:],1,axis=1)
    f[6,:,:] = np.roll(f[6,:,:],-1,axis=0)
    f[6,:,:] = np.roll(f[6,:,:],1,axis=1)

    f[1,:,:]=np.roll(f[1,:,:],1,axis=0)
    f[5,:,:]=np.roll(f[5,:,:],1,axis=0)
    f[5,:,:]=np.roll(f[5,:,:],1,axis=1)
    
    f[4,:,:] = np.roll(f[4,:,:],-1,axis=1)
    f[8,:,:]= np.roll(f[8,:,:],1,axis=0)
    f[8,:,:]= np.roll(f[8,:,:],-1,axis=1)
  
    f[3,:,:] = np.roll(f[3,:,:],-1,axis=0)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=0)
    f[7,:,:] = np.roll(f[7,:,:],-1,axis=1)

    # Left  i=0, adiabatic  
    f[:,0,:] = f[:,1,:]
    # Right,i=n, adiabatic 
    f[:,n,:] = f[:,n-1,:]
    # BOTTOM: j=m (inverting top and bottom definitions)
    f[6,:,0] = w[6]*T_base + w[8]*T_base - f[8,:,0]
    f[5,:,0] = w[5]*T_base + w[7]*T_base - f[7,:,0]
    f[2,:,0] = w[2]*T_base + w[4]*T_base - f[4,:,0]
    #f[1,:,0] = w[1]*T_base + w[3]*T_base - f[3,:,0]
    #f[0,:,m] = 0.0
    # TOP j=0  
    f[7,:,m] = w[7]*T_top + w[5]*T_top - f[5,:,m]
    f[4,:,m] = w[4]*T_top + w[2]*T_top - f[2,:,m]
    f[8,:,m] = w[8]*T_top + w[6]*T_top - f[6,:,m]
    #f[0,:,0] = 0.0


    T = f[0,:,:] + f[1,:,:] + f[2,:,:] + f[3,:,:] + f[4,:,:] +f[5,:,:] + f[6,:,:]+f[7,:,:]+f[8,:,:]
    LB_struct_2D['T'] = T
    LB_struct_2D['f'] = f
    return LB_struct_2D

  def update_time(self,age,LB_struct_2D):
    age_sec = age*60*60*24*365.25*1e6
    dx = LB_struct_2D['dx']
    dt = LB_struct_2D['dt']
    kappa = LB_struct_2D['kappa']
    mstep = int(age_sec/dt)
    ck=dx/dt
    csq=ck*ck
    omega=1.0/(3*kappa/(dt*csq)+0.5) 
    LB_struct_2D['omega']=omega
    LB_struct_2D['mstep']=mstep
    return LB_struct_2D
    
  def init_linear_temp(self, LB_struct_2D, T_top, T_bottom):
    T =  LB_struct_2D["T"]
    m =  LB_struct_2D["m"]
    T1 = np.linspace(1500,0,m+1)
    for j in np.arange(0,m+1,1):
        T[:,j] = T[:,j]*T1[j]
    LB_struct_2D["T"]=T
    return  LB_struct_2D

  def init_slabs_T(self,  LB_struct_2D, plate_age1, plate_age2, distance,Tinit):
    K = LB_struct_2D["kappa"]
    m =  LB_struct_2D["m"]
    n =  LB_struct_2D["n"]
    dy =  LB_struct_2D["dy"]
    dx =  LB_struct_2D["dx"]
    T =  LB_struct_2D["T"]
    T = np.ones_like(T)
    y = np.arange(0,dy*(m+1),dy)
    T1 = np.flip((Tinit*special.erf(y/(2*np.sqrt(K*plate_age1)))))
    T2 = np.flip((Tinit*special.erf(y/(2*np.sqrt(K*plate_age2)))))
    for i in np.arange(0,n+1,1):
        dist = dx*i
        if dist < distance:
            for j in np.arange(0,m+1,1):
                T[i,j] = T[i,j]*T1[j]
        else:
            for j in np.arange(0,m+1,1):
                T[i,j] = T[i,j]*T2[j]

    LB_struct_2D["T"] = T

    w=np.zeros(9)
    w[0]=4./9.
    w[1]=w[2]=w[3]=w[4]=1./9.
    w[5]=w[6]=w[7]=w[8]=1./36.

    f=LB_struct_2D['f']
    f[0,:,:] =  w[0]*T
    f[1,:,:] =  w[1]*T
    f[2,:,:] =  w[2]*T
    f[3,:,:] =  w[3]*T
    f[4,:,:] =  w[4]*T
    f[5,:,:] =  w[5]*T
    f[6,:,:] =  w[6]*T
    f[7,:,:] =  w[7]*T
    f[8,:,:] =  w[8]*T
    LB_struct_2D['f']=f

    thickness = 0
    for  j in np.arange(0,m+1,1):
        depth = dy * j
        if T1[j] < 1300:
            thickness = depth
            break
    thickness = dx*m+1 - thickness
    LB_struct_2D['plate_age'] = plate_age1

    return  LB_struct_2D, thickness

  def LB_D2Q9_init_subduction(self,dt,m,n,dx,dy,T0,H,kappa,mstep):
            
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
    T=np.ones((n+1,m+1))*T0
    #global vx
    vx=np.zeros((n+1,m+1))
    #global vy
    vy=np.zeros((n+1,m+1))

    vx[1:n-1,m]=0.
    vy[1:n-1,m]=0.


    w=np.zeros(9)
    w[0]=4./9.
    w[1]=w[2]=w[3]=w[4]=1./9.
    w[5]=w[6]=w[7]=w[8]=1./36.

    cx = np.array([0.0,1.0,0.0,-1.0,0.0,1.0,-1.0,-1.0,1.0])
    cy = np.array([0.0,0.0,1.0,0.0,-1.0,1.0,1.0,-1.0,-1.0])

    f[0,:,:] =  w[0]*T
    f[1,:,:] =  w[1]*T
    f[2,:,:] =  w[2]*T
    f[3,:,:] =  w[3]*T
    f[4,:,:] =  w[4]*T
    f[5,:,:] =  w[5]*T
    f[6,:,:] =  w[6]*T
    f[7,:,:] =  w[7]*T
    f[8,:,:] =  w[8]*T
    
    ck=dx/dt
    csq=ck*ck
    omega=1.0/(3*kappa/(dt*csq)+0.5) 


    #omegaV=1.0/(3.*kappa+0.5) 
    #T = np.ones((m+1))*Tinit
    #   self,dt,dx,mstep,m,Tinit,TSurf,H

    LB_struct = {
      'dt': dt,
      'mstep': mstep,
      'm': m,
      'n':n,
      'dx':dx,
      'dy':dy,
      'vx':vx,
      'vy':vy,
      'vx0':vx,
      'vy0':vy,
      'cx':cx,
      'cy':cy,
      'T': T,
      'T_top':0.0,
      'T_base':T0,
      'H': H,
      'w':w,
      'omega': omega,
      'f':f,
      'feq':feq,
      'x':X,
      'y':Y,
      'kappa':kappa,
      'plate_age':60.0,
    }
    return LB_struct

  def define_subduction_velocity_sharp(self,LB_struct_2D,thickness,velocity,angle,distance):
    K = LB_struct_2D["kappa"]
    m =  LB_struct_2D["m"]
    n =  LB_struct_2D["n"]
    dy =  LB_struct_2D["dy"]
    dx =  LB_struct_2D["dx"]
    T =  LB_struct_2D["T"]
    #vx =  LB_struct_2D["vx"]
    #vy =  LB_struct_2D["vy"]
    vx = np.zeros_like(LB_struct_2D["vx"])
    vy = np.zeros_like(LB_struct_2D["vy"])
    tot_depth = dy * (m+1)
    angle = np.deg2rad(angle)
    dh = thickness / np.cos(angle)
    sub_vx = velocity*np.sin(angle)
    sub_vy = velocity*np.cos(angle)
    buffer = distance - (thickness*np.tan(angle))
    for i in np.arange(n,0,-1):
        dist = dx*i
        if dist < buffer:
            for j in np.arange(0,m+1,1):
                depth = tot_depth - dy * j
                if depth < thickness:
                    #print(i,j,dist,tot_depth,dy*j,depth,thickness)
                    vx[i,j] = velocity
                    vy[i,j] = 0.
        elif (dist >= buffer)&(dist < distance):
            db = -distance + dist
            d1 = db *np.tan(angle)
            d2 = d1 + thickness/np.cos(angle)
            for j in np.arange(0,m+1,1):
                depth = tot_depth - dy * j
                if (depth > db/np.tan(angle))&(d2 > thickness):
                        if ( (depth > d1) & (depth < d2)):
                            #print(i,j,dist,depth,thickness,d1,d2)
                            vx[i,j] = sub_vx
                            vy[i,j] = -sub_vy
                elif (depth < thickness):
                            vx[i,j] = velocity
                            vy[i,j] = 0.
        else:
            db = dx*i - distance
            d1 = db *np.tan(angle)
            d2 = d1 + thickness/np.cos(angle)
            for j in np.arange(0,m+1,1):
                depth = tot_depth - dy * j
                if ( (depth > d1) & (depth < d2)):
                    #print(i,j,dist,depth,thickness,d1,d2)
                    vx[i,j] = sub_vx
                    vy[i,j] = -sub_vy
    LB_struct_2D["vx"] = vx
    LB_struct_2D["vy"] = vy
    LB_struct_2D["vx0"] = vx
    LB_struct_2D["vy0"] = vy

    return LB_struct_2D

  def define_subduction_velocity_smooth(self,LB_struct_2D,thickness,velocity,angle,distance):
    K = LB_struct_2D["kappa"]
    m =  LB_struct_2D["m"]
    n =  LB_struct_2D["n"]
    dy =  LB_struct_2D["dy"]
    dx =  LB_struct_2D["dx"]
    T =  LB_struct_2D["T"]
    X =  LB_struct_2D["x"]
    Y =  LB_struct_2D["y"]
    vx = np.zeros_like(LB_struct_2D["vx"])
    vy = np.zeros_like(LB_struct_2D["vy"])

    tot_depth = dy * (m+1)
    angle = np.deg2rad(angle)
    dh = thickness / np.cos(angle)
    sub_vx = velocity*np.sin(angle)
    sub_vy = velocity*np.cos(angle)
    buffer = distance - (thickness*np.tan(angle))

    x1 = distance
    y1 = np.max(Y) - thickness * 1.5
    r1 = 1.0*thickness * 1.5
    r2 =  0.5*thickness*1.0
    phi = (np.arctan2(Y-y1,X-x1) ) #np.rad2deg 

    circle_filter = (( (X - x1)**2 + (Y-y1)**2 <= r1**2)&((X - x1)**2 + (Y-y1)**2 > r2**2  ) )  & ( (X >= x1)&(Y>y1)&(phi > 0)&(phi > angle))

    dist_filter = (X <= distance)&(Y>(np.max(Y)-thickness))

    xphi1 = x1 + np.sin(angle)*r1
    xphi2 = x1 + np.sin(angle)*r2
    yphi1 = y1 + np.cos(angle)*r1
    yphi2 = y1 - np.cos(angle)*r2

    print("yphi",yphi1/1e3,np.rad2deg(angle),"r",r1/1e3,"y1",y1/1e3)
    d1 = yphi1 - (X-xphi1) *np.tan(angle)
    d2 = d1 - 1.0*thickness/np.cos(angle)
    print("d1",d1)

    #
    deep_filter = ((X > distance)&(Y<d1)&(Y>d2)&(phi <= angle))  #& (Y < yphi1+np.tan(angle)*(X-xphi1)) & (Y > yphi2 +np.tan(angle)*(X-xphi2))

    vx[dist_filter] = velocity
    vy[dist_filter] = 0.0

    vx[circle_filter] = velocity*np.sin(phi[circle_filter])
    vy[circle_filter] =  -velocity*np.cos(phi[circle_filter])

    vx[deep_filter] = sub_vx
    vy[deep_filter] = -sub_vy

    LB_struct_2D["vx"] = vx
    LB_struct_2D["vy"] = vy
    LB_struct_2D["vx0"] = vx
    LB_struct_2D["vy0"] = vy

    return LB_struct_2D


  def LB_D2Q9_Subduction(self,LB_struct_2D):
    n = LB_struct_2D['n']
    m = LB_struct_2D['m']
    vx = LB_struct_2D['vx']
    vy = LB_struct_2D['vy']
    cx = LB_struct_2D['cx']
    cy = LB_struct_2D['cy']
    T = LB_struct_2D['T']
    dt = LB_struct_2D['dt']
    w = LB_struct_2D['w']
    H = LB_struct_2D['H']
    omegaV = LB_struct_2D['omega']
    f = LB_struct_2D['f']
    T_base = LB_struct_2D['T_base']
    T_top = LB_struct_2D['T_top']
    feq = np.zeros_like(f)
    Y = LB_struct_2D['y']
    K =  LB_struct_2D['kappa']

    #omega=omegaV
    ck = LB_struct_2D['dx']/LB_struct_2D['dt']
    #feq = self.loopy_loop(n,m,vx,vy,cx,cy,T,w)
    feq[0,:,:] = w[0]*T
    feq[1,:,:] = w[1]*T*(1. + 3.*vx/ck)
    feq[2,:,:] = w[2]*T*(1. + 3.*vy/ck)
    feq[3,:,:] = w[3]*T*(1. - 3.*vx/ck)
    feq[4,:,:] = w[4]*T*(1. - 3.*vy/ck)
    feq[5,:,:] = w[5]*T*(1. + 3.*(vx + vy)/ck)
    feq[6,:,:] = w[6]*T*(1. + 3.*(-vx + vy)/ck)
    feq[7,:,:] = w[7]*T*(1. - 3.*(vx + vy)/ck)
    feq[8,:,:] = w[8]*T*(1. + 3.*(vx - vy)/ck)

    #for i,z in np.ndenumerate(w):
    #    feq[i,:,:] = w[i]*T
    f = (1.0 - omegaV)*f + omegaV*feq + dt*0.5*H

    #Collision step
    #f[0,:,:] = (1.0 - omegaV)*f[0,:,:] + omegaV*feq[0,:,:]  + dt*0.5*H
    #f[1,:,:] = (1.0 - omegaV)*f[1,:,:] + omegaV*feq[1,:,:]  + dt*0.5*H
    #f[2,:,:] = (1.0 - omegaV)*f[2,:,:] + omegaV*feq[2,:,:]  + dt*0.5*H
    #f[3,:,:] = (1.0 - omegaV)*f[3,:,:] + omegaV*feq[3,:,:]  + dt*0.5*H
    #f[4,:,:] = (1.0 - omegaV)*f[4,:,:] + omegaV*feq[4,:,:]  + dt*0.5*H
    #f[5,:,:] = (1.0 - omegaV)*f[5,:,:] + omegaV*feq[5,:,:]  + dt*0.5*H
    #f[6,:,:] = (1.0 - omegaV)*f[6,:,:] + omegaV*feq[6,:,:]  + dt*0.5*H
    #f[7,:,:] = (1.0 - omegaV)*f[7,:,:] + omegaV*feq[7,:,:]  + dt*0.5*H
    #f[8,:,:] = (1.0 - omegaV)*f[8,:,:] + omegaV*feq[8,:,:]  + dt*0.5*H

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
  

    # Left  i=0, adiabatic  
    # Need to make error function!
    #  T1 = np.flip((Tinit*special.erf(y/(2*np.sqrt(K*plate_age1)))))
    plate_age1 = LB_struct_2D['plate_age']
    Terr = np.flip ( ((T_base*special.erf(Y[0,:]/(2*np.sqrt(K*plate_age1))))) )
    #print("Terr",T_base,K,plate_age1,Terr)
    #f[:,0,:] = f[:,1,:]  # Adiabatic
    #LHS
    #i=0
    #for Ter in Terr:
    f[6,0,:m] = w[6]*Terr[:m] + w[8]*Terr[:m] - f[8,0,:m]
    f[3,0,:m] = w[3]*Terr[:m] + w[1]*Terr[:m] - f[1,0,:m]
    f[7,0,:m] = w[7]*Terr[:m] + w[5]*Terr[:m] - f[5,0,:m]
    #    i += 1

    # Right,i=n, adiabatic 
    f[:,n,:] = f[:,n-1,:]

    # TOP
    f[6,:,0] = w[6]*T_base + w[8]*T_base - f[8,:,0]
    f[5,:,0] = w[5]*T_base + w[7]*T_base - f[7,:,0]
    f[2,:,0] = w[2]*T_base + w[4]*T_base - f[4,:,0]
    #f[1,:,0] = w[1]*T_base + w[3]*T_base - f[3,:,0]
    #f[0,:,m] = 0.0
    # BOTTOM  
    f[7,:,m] = w[7]*T_top + w[5]*T_top - f[5,:,m]
    f[4,:,m] = w[4]*T_top + w[2]*T_top - f[2,:,m]
    f[8,:,m] = w[8]*T_top + w[6]*T_top - f[6,:,m]
    #f[0,:,0] = 0.0


    T = f[0,:,:] + f[1,:,:] + f[2,:,:] + f[3,:,:] + f[4,:,:] +f[5,:,:] + f[6,:,:]+f[7,:,:]+f[8,:,:]
    #print(T)
    #print(f)
    #T[:,m] = f[0,:,m] + f[1,:,m] + f[3,:,m] + 2.0*(f[2,:,m] + f[6,:,m] + f[5,:,m])

    #usum = f[0,:,:]*cx[0] + f[1,:,:]*cx[1] + f[2,:,:]*cx[2] + f[3,:,:]*cx[3] + f[4,:,:]*cx[4] +f[5,:,:]*cx[5] + f[6,:,:]*cx[6] +f[7,:,:]*cx[7]+f[8,:,:]*cx[8]
    #vsum = f[0,:,:]*cy[0] + f[1,:,:]*cy[1] + f[2,:,:]*cy[2] + f[3,:,:]*cy[3] + f[4,:,:]*cy[4] +f[5,:,:]*cy[5] + f[6,:,:]*cy[6] +f[7,:,:]*cy[7]+f[8,:,:]*cy[8]
    #vx = usum/T
    #vy = vsum/T
    LB_struct_2D['T'] = T
    LB_struct_2D['f'] = f
    return LB_struct_2D




