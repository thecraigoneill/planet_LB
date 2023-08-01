import planet_LB as p_LB
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def test1():
    # Initialise constants and object
    M2S = p_LB.Myr2s
    pLB = p_LB.planet_LB()

    #Time discretisation
    age = 50 #Age in Myrs
    tstep=2000 #Number of timesteps
    dt = age*M2S/tstep

    #Mesh discretisation
    m=400  # Number of x divisions
    dx=500.0 #Size of grid spacing


    #Initial/boundary and heating properties
    H=0. # internal heat production
    Tinit = 1300 # Initial temperature
    TSurf = 0.0 #Surface temperature
    K=1e-6
    k=3
    Qbase=0

    # Simulation 1

    LB = pLB.init_1D_LB(dt,dx,tstep,m,Tinit,TSurf,H,Qbase,K,k)
    T=pLB.LB1D(LB)
    #print(LB['T'])
    print(T)

def test2():
    M2S = p_LB.Myr2s
    pLB = p_LB.planet_LB()

    #Time discretisation
    age = 15 #Age in Myrs
    dt = 10000.0*M2S/1e6 # in sec
    mstep=1

    #Mesh discretisation
    m=300  # Number of x divisions
    n=300
    #Scale problem###################################################
    depth = 30e3
    K_d=1.1538461538461538e-06

    dim_side = K_d*age*M2S/(depth)**2

    N=300
    K_nd=1
    # then time steps M
    M = int(dim_side*(N**2)/K_nd)
    print("timesteps",M)

    d_star = depth/N
    mstep=M
    dx=1 
    dy=1
    dt=1
    ck=dx/dt
    csq=ck*ck 
    Tinit = 600 # Initial temperature
    TSurf = 0.0 #Surface temperature
    Qbase = 0 # Not used

    dens1=2600
    dens2=2950

    k1=3.0
    k2=3.0
    C=1000  
    K1 = k1/(dens1*C)
    K2 = k1/(dens2*C)
    kappa=K1 #placeholder

    Z1=20e3
    Z2=30e3
    Hnd = ((d_star**2)/(K1*(1/1.153846)))/Tinit  #Scaling factor for heat production
    A1d= 9.6e-10
    A2d=A1d
    A1= Hnd * A1d 
    A2=A1

    LBS = pLB.LB_D2Q9_init_ThermHF(dt,m,n,dx,dy,mstep,TSurf,Tinit,A2,Qbase,kappa,k1)
    y=LBS['y']

    H = np.ones_like(y) * A2  
    kappa = np.ones_like(y) * K1
    H[ (y) >= Z1] = A1
    kappa[(y)>=Z1] = K2 
    kappa *= 1e6
    LBS = pLB.LB_D2Q9_init_ThermHF(dt,m,n,dx,dy,mstep,TSurf,Tinit,H,Qbase,kappa,k1)
    X=LBS['x']
    T=np.ones_like(LBS['T'])
    T1 = T * np.linspace(Tinit,0,m+1)
    LBS = pLB.reinit_T_f(LBS,T1)

    mstep=LBS['mstep']
    print(mstep)
    T=LBS['T']
    f=LBS['f']
    #ONE STEP
    LBS =  pLB.LB_D2Q9_T_HF(LBS,0,0)
    T=LBS['T']
    print(T)


def test3():
    M2S = p_LB.Myr2s
    pLB = p_LB.planet_LB()

    uo=0.10
    rhoo=5.0

    m=100
    n=100

    dist_x = 1.0
    dist_y = 1.0
    # Backing out increments
    dx = dist_x/n
    dy= dist_y/m

    alpha=0.01 
    H=0
    mstep=1

    dt=1.0

    lb_s = pLB.LB_D2Q9_init_Lid(dt,m,n,dx,dy,rhoo,uo,H,alpha,mstep)
    # Test one step
    rho, vx, vy, f =  pLB.LB_D2Q9_V(lb_s)
    print(vx)

    
