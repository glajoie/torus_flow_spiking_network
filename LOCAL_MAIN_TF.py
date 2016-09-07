#Jan 2014 -- Guillaume Lajoie


#Script to perform a LOCAL (on a single computer) simulation of theta-neuron networks coded in Torus_Flow_Solver.pyx

#This script uses CL_caller_TF.py using systems cammand calls and outputs the results of each run in a separate
#folder all contained in an "output/" folder wich must be created beforehand.

#This script needs the module "architecture_TF.py" that contains a function
#to create connectivity matrix and other randomnly generated network parameters.
#The architecture file to be used is saved in a pickle file: "PARAMS_<run_name>.pkl"

#============================
#OUTPUT: 

#Output files are saved in ".m" MATLAB format. This code is setup to specify a parameter
#grid for epsilon (input strength to each neuron), eta (excitability of each neuron) and
#c (correlation of input between neurons). See "RUN PARAMETERS" section below.

#For each parameter grid point there will be num_IC 
#randomly chosen initial condtitions evolved. Therefore, there will be [num_IC X num_epsilon 
#X num_eta X num_c] sub-runs submitted serially (one after the other).
#Each sub-run's data will be stored in an independent sub-folder:
# "<run_name>_<IC-index>_<epsilon-index>_<eta-index>_<c-index>/data.m"
#============================


#================================================================================
#Importing required modules
#================================================================================
import numpy as np
import pickle as pkl
import sys
import scipy.io as io
import os
from subprocess import call as call
from architecture_TF import arch_maker

#================================================================================
#Run name and saving directories
#================================================================================
run_name='TF_test'

basedir='./' #don't touch for local runs
output_dir='output'
if os.path.isdir(basedir+output_dir)==False: #create output directory if not present
	os.mkdir(basedir+output_dir)


#================================================================================
#Switchboard: decides what type of run to do. All must be 0 or 1.
#================================================================================
traj_switch=1 #saves trajectories of N_rec neurons at the sampling resolution defined below
spike_times_switch=1 #saves spike time of N_rec neurons
LE_switch=0 #compute first k_LE_spectrum LEs
FTLE_switch=0 #WORKS ONLY IF LE_switch==1 and returns full LE basis and Le vector norms evolution 
pseudo_LE_switch=0 #Test to see if first LE is OK (better leave at 0)
drive_save_switch=1 #only save from first sim if static_drive_switch==1 (saves space)
static_drive_switch=1 #Tells that the same input (quenched noise) must presented on each subrun.

#================================================================================
#Solver parameters
#================================================================================
t_span=10. #length of sim
t_record=0. #time to start recording
dt=.005 #0.0025 is very good and 0.005 works fine
dt_sample=5*dt #time to sample and rescale LE computation
sample_save_increment=1 #number of dt_samples between saves (of trajs also). Must be an int!

N_rec=10 #number of neurons to record
k_LE_spectrum=3 #number of LEs to compute if appropriate switch is on.

#================================================================================
#RUN PARAMETERS
#================================================================================
num_IC=3 #number of random initial conditions for each parameter set

#Input strength epsilon>=0
num_epsilon=1 #drive strength
epsilon_min=0.5
epsilon_max=0.5

#Neural excitability. Typically, -1<=eta<=1.
num_eta=1 #excitability
eta_min=-0.5
eta_max=-0.5

#Input correlation across neurons: 0<=c<=1. (c=0:uncorrelated, c=1:common input)
num_c=1 #correllation of inputs across network cells
c_min=0.
c_max=0.

mu_noise=0 #strength of independent noise

N=500 #size of network
N_E=0.8*N #number of excitatory neurons
N_D=N #number of neurons receiving a drive
K=20 #connectivity "in-degree"

#================================================================================
#MISC
#================================================================================
print_progress=1 #prints simulation progress at command line
hist_bin_num=10 #number of bins for invariant measure histogram
xi=0.0005 #perturbation for pseudo traj if switch is on
pseudo_dt=2*dt #time step between renormalization of pseudo traj

b=.05 #width of synaptic function
A_dial=0.35 #do not touch
i_eta_offset=-0.1 #do not touch
eta_sample_width=0.01 #do not touch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#================================================================================
#Calling architecture making function and creating dictionary of parameters
#================================================================================
architecture=arch_maker(N,N_E,N_D,K,N_rec,num_epsilon,epsilon_min,epsilon_max,num_eta,eta_min,eta_max,num_c,c_min,c_max,num_IC,mu_noise,A_dial,i_eta_offset,eta_sample_width,b)
param_dict={'t_span': t_span, 'dt': dt, 'dt_sample': dt_sample, 't_record':t_record,
  'sample_save_increment': sample_save_increment, 'traj_switch': traj_switch, 'spike_times_switch': spike_times_switch,
   'LE_switch': LE_switch, 'FTLE_switch': FTLE_switch, 'drive_save_switch':drive_save_switch, 'pseudo_LE_switch':pseudo_LE_switch,'k_LE_spectrum':k_LE_spectrum,
   'hist_bin_num':hist_bin_num, 'xi':xi, 'pseudo_dt':pseudo_dt,'architecture':architecture,'print_progress':print_progress}
param_dict.update(architecture) #mergin 'architecture' dictionary into 'param_dict'

 
#================================================================================
#Creating bogus RNG seed for non static drive
#================================================================================
if static_drive_switch==0:
	seedList=np.array([0]) 
	param_dict['seedList']=seedList

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#================================================================================
#Saving parameter files
#================================================================================
pkl.dump(param_dict,open('PARAMS_'+run_name+'.pkl','wb')) #pickle format to be read by 'CL_caller_TF.py'
io.savemat(output_dir+'/PARAMS_'+run_name,param_dict) #matlab format 
print('---->Parameter files created and saved<----')
print('---->Now compiling solver module<----')

#================================================================================
#Compiling cython solver module
#================================================================================
call('python setup_TF.py build_ext --inplace', shell=True)
print('---->Solver compiled successfully, ready to rock!!!<----')

#================================================================================
#Solver calls for LOCAL machine
#================================================================================
print('---->Starting runs...<----')
save_flag=0
if drive_save_switch==1:
	save_flag=1
tak=1
for IC_index in range(num_IC):
	for epsilon_index in range(num_epsilon):
		for eta_index in range(num_eta):
			for c_index in range(num_c):
				print('---->Ongoing run '+str(tak)+'/'+str(num_IC*num_eta*num_epsilon)+'<----')
				tak=tak+1
				call('python CL_caller_TF.py '+run_name+' '+output_dir+' '+basedir+' '+str(IC_index)+' '+str(epsilon_index)+' '+str(eta_index)+' '+str(c_index)+' '+str(save_flag), shell=True)
				# call('python CL_caller_TF.py '+run_name+' '+output_dir+' ./ '+str(IC_index)+' '+str(epsilon_index)+' '+str(eta_index)+' '+str(save_flag), shell=True)
				save_flag=0
print('--->All Done !<---')



