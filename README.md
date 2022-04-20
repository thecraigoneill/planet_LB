# planet_LB
A python distribution for lattice-boltzmann modelling of planetary processes

Currently has 1D (D1Q2) and 2D (D2Q9) solvers for the advection-diffusion problems, from plate cooling and contact metamorphism, to subduction zones.

Install:

pip install planet_LB

See the ipynb for usage and examples, and some 1D and 2D benchmarks.

Here's an example of the cooling of a thrust model:

![Fowler_thrust](https://user-images.githubusercontent.com/30849698/164211427-734a2e79-ef8e-466d-b2a9-016c539c7b31.png)

Or, if subduction floats (sinks?) your boat, here's one we prepared earlier: 

![subduction](https://user-images.githubusercontent.com/30849698/164226119-fc9efb78-9431-4759-8e3d-37d3f07e10af.png)

Full code in the ipynb directory. Enjoy. Or not. Up to you, really. 


# A few worthy references

Fowler, C.M.R., The solid Earth: An introduction to global geophysics (2004). (A great entry-level drug into the broader world of geodynamics. Some wonderful examples from her seminal 1980's papers on heat flow - borrowed from in examples here).

Turcotte, D., Schubert, G., Geodynamics (2001). (Gold standard geodynamics textbook).

Mohamad, A.A. Lattice Boltzmann Method: Fundamentals and Engineering Applications with Computer Codes (2011). (A nice and very practical introduction to Lattice Boltzmann methods).
