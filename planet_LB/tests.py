import planet_LB as p_LB
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

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

# Simulation 1

T,x = pLB.init_1D_LB(dt,dx,tstep,m,Tinit,TSurf,H)

# Simulation II - an oceanic crust made up of granite in the top 10km, with mantle heat production beneath that:
# Heat production. 
H1 = np.ones_like(x) * 1e-12  #Note H is now being used as an array - can be constant or x-like array
H1[ x <= 10e3] = 9.7e-9

# Note H(W/kg)*dens = W/m3 = qg.
# Source S = qg/dens*C --> H*dens/dens*Cp --> H/Cp
Cp = 1000.0
H1 /= Cp

T1,x1 = pLB.init_1D_LB(dt,dx,tstep,m,Tinit,TSurf,H1)

