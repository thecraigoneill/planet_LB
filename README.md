# planet_LB
A Python distribution for lattice-Boltzmann modelling of planetary processes

Currently has 1D (D1Q2) and 2D (D2Q9) solvers for the advection-diffusion problems, from plate cooling and contact metamorphism to subduction zones.


# Installation:

The preferred procedure for installation is using pip as follows:

    pip install planet_LB

You can install from the git repository download as well, using the command:

    pip install .

from the main directory.


# Requirements

The library has minimal requirements, using Python 3.+ and the basic Python libraries NumPy (tested on 1.7.2), SciPy (tested on 1.3.1), and Matplotlib (tested on version 3.4.2), and the code loops are accelerated using Numba (tested on 0.45.1). 

# Examples

See the Jupyter notebook for usage, examples, and 1D and 2D benchmarks.

Here's an example of the cooling of a thrust model:

![Fowler_thrust](https://user-images.githubusercontent.com/30849698/164211427-734a2e79-ef8e-466d-b2a9-016c539c7b31.png)

Or, if subduction floats (sinks?) your boat, here's one we prepared earlier: 

![subduction](https://user-images.githubusercontent.com/30849698/164226119-fc9efb78-9431-4759-8e3d-37d3f07e10af.png)

Full code in the ``ipynb`` directory. Enjoy. Or not. Up to you, really. 


# Community contributions
We totally welcome community contributions to the code base. Contact Craig with specific ideas/requests, to confirm if the functionality exists. If not, you are welcome to fork the repo, clone it locally (git clone <link to repo>), create a new branch (git checkout -b new-user-contribution), and make modifications as you wish. To send these back to the main branch, you need to stage (eg. git add .\Contributors.md), commit (eg. git commit -m “Add XYZ to Contributors List”), and then push the changes back to your fork. Then create a pull request to send these back to the main for checking (and perhaps drop me a note about the functionality changes).  More info on the process here: https://dev.to/codesphere/how-to-start-contributing-to-open-source-projects-on-github-534n


# A few worthy references

- Fowler, C.M.R., The solid Earth: An introduction to global geophysics (2004). (A great entry-level drug into the broader world of geodynamics. Some wonderful examples from her seminal 1980's papers on heat flow - borrowed from in examples here).

- Turcotte, D., Schubert, G., Geodynamics (2001). (Gold standard geodynamics textbook).

- Mohamad, A.A. Lattice Boltzmann Method: Fundamentals and Engineering Applications with Computer Codes (2011). (A nice and very practical introduction to Lattice Boltzmann methods).
