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

    # Simulation 1

    LB = pLB.init_1D_LB(dt,dx,tstep,m,Tinit,TSurf,H)
    T=pLB.LB1D(LB)
    #print(LB['T'])
    print(T)

