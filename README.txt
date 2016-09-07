Guillaume Lajoie January 2014 (http://faculty.washington.edu/glajoie/wordpress/)

The Torus_Flow (TF) suite is a python/cython code implemented to simulate a network of theta-neurons of arbitrary size receiving white noise inputs. The code can record total trajectories of any number of neurons as well as spike times and estimates of Lyapunov Spectra of any truncated size.

#####################################################
More details about the network model can be found in:

-Chaos and reliability in balanced spiking networks with temporal drive, Guillaume Lajoie, Kevin K. Lin and Eric Shea-Brown, Phys. Rev. E, (2013), Vol. 8, No. 5, pp. 052901

-Structured chaos shapes spike-response noise entropy in balanced neural networks, Guillaume Lajoie, Jean-Philippe Thivierge and Eric Shea-Brown, Frontiers in Computational Neuroscience, (2014),  8:123
#####################################################

DISCLAIMER: This software was design for specialized scientific computation only and is not meant as a deployable product. As a result, some warnings may appear at the time of compilation and some parameter settings may lead to errors.

REQUIREMENTS: Python3 and Cython must be installed. For an easy way to do this, install the latest conda distribution and make sure the Cython package is installed. MATLAB as the output data is saved in ".m" format.

USE: The main script calling the code is entitled LOCAL_MAIN_TF.py. It is setup to launch multiple runs going through multiple parameter sets and initial conditions. The code itself is well documented. 


FILES NEEDED in base directory: 
1.Torus_Flow_Solver.pyx (the cython file of the solver)
2.MersenneTwister.h (the header for the random number generator)
3.setup_TF.py (the setup file for compiling the solver)
4.CL_caller_TF.py (command-line calling script)
5.architecture_TF.py (python module containing a function that returns parameters such as connectivity matrix and single neuron parameters)
6.launcher.sge (ONLY FOR LONESTAR RUNS)
7.LOCAL / LONESTAR ..._MAIN_TF.py (the main script where run details are set)

QUICK START:
There are two possible ways for quick use:
(1) in terminal, navigate to directory containing code and type "python LOCAL_MAIN_TF.py"
(2) run the python notebook "Basic TORUS_FLOW tutorial.ipynb" for interactive explanations.
The output of simulations from each parameter set will be stored in a separate folder in "output/" in a MATLAB format. Each folder will have the run_name followed by four numbers identifying the parameter set form the sub run. See LOCAL_MAIN_TF.py for more details.