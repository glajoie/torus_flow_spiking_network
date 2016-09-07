#Jan 2014 Guillaume Lajoie

#Script to be called from command line to start a solver run of 'Torus_Flow_Solver.pyx'

#A pickled file called 'PARAMS_<run_name>.pkl' is needed in the base directory.

#This script is meant to be called from the '_MAIN_TF.py' script which creates the parameter files.


#TAKES THE FOLLOWING AS ARGUMENT :
#=======================================
#=======================================
#args=['sub_run_name', 'output_dir', run_nbr] ------
#EXAMPLE CALL:  
#python CL_caller.py sub_run_name output_dir base_dir IC_index epsilon_index eta_index c_index save_flag
#=======================================
#=======================================

# Import necessary packages:
from Torus_Flow_Solver import N_torus_integrate
import time as tim
import numpy as np
import pickle as pkl
import sys
import scipy.io as io
from subprocess import call as call


#Taking arguments from command line -> in a float list args
run_name = sys.argv[1]
output_dir=sys.argv[2]
base_dir=sys.argv[3]
IC_index=int(sys.argv[4])

#GRID STUFF&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#These are for parameter searches
#Extracting index for grid parameters
epsilon_index=int(sys.argv[5])
eta_index=int(sys.argv[6])
c_index=int(sys.argv[7])
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
save_flag=int(sys.argv[8]) #flag to save save_times and drive_container for this run


#Loading parameters
params=pkl.load(open(base_dir+'PARAMS_'+run_name+'.pkl','rb'))


traj_switch=params['traj_switch']
spike_times_switch=params['spike_times_switch']
LE_switch=params['LE_switch']
FTLE_switch=params['FTLE_switch']
drive_save_switch=params['drive_save_switch']
pseudo_LE_switch=params['pseudo_LE_switch']

b=params['b']
t_span=params['t_span']
t_record=params['t_record']
dt=params['dt']
dt_sample=params['dt_sample']
sample_save_increment=params['sample_save_increment'] 
hist_bin_num=params['hist_bin_num']
print_progress=params['print_progress']

eta=params['eta'][eta_index] #GRID stuff &&&&&&&&&&&
epsilon=params['epsilon'][epsilon_index] #GRID stuff &&&&&&&&&&&
c=params['c_vector'][c_index] #GRID stuff &&&&&&&&&&&&
IC_array=params['IC']
A=params['A']
N=params['N']
N_rec=params['N_rec']
rec_index=params['rec_index']
N_D=params['N_D']
D_index=params['D_index']
mu_noise=params['mu_noise']

seedList=params['seedList']

k_LE_spectrum=params['k_LE_spectrum']

#IC=IC_array[IC_index,:]
IC=IC_array[IC_index,:]

#pseudo traj compute params
xi=params['xi']
pseudo_dt=params['pseudo_dt']

sub_run_name=run_name+'_'+str(IC_index)+'_'+str(epsilon_index)+'_'+str(eta_index)+'_'+str(c_index)


#Creating output folder for this run++++++
call('mkdir '+output_dir+'/Matlab_output_'+sub_run_name, shell=True)
#+++++++++++++++++++++++++++++++++++++++++


# Start timer:
tBegin = tim.mktime(tim.localtime())

#CALLING SOLVER--------------------------

out_str, out= N_torus_integrate(N, N_rec, rec_index, D_index, eta, A, epsilon, b, t_span, dt, dt_sample, sample_save_increment, hist_bin_num,  IC, c, seedList, xi, pseudo_dt, mu_noise, k_LE_spectrum,
 traj_switch, spike_times_switch, LE_switch, FTLE_switch, pseudo_LE_switch, save_flag, print_progress)

#creating output dictionnary
data_dict={}
for it in range(len(out_str)):
	if out_str[it]=='spike_times_container': #extracting spike times
		spike_times_container=out[it]
		spike_times_container_size=np.shape(spike_times_container)
		spike_times_cell=np.ndarray(spike_times_container_size[0], dtype=np.object)
		for it in range(spike_times_container_size[0]):
			index=np.nonzero(spike_times_container[it,:]>0)
			if index:
				spike_times_cell[it]=spike_times_container[it,index]
			elif not index:
				spike_times_cell[it]=[]
		data_dict['spike_times_cell']=spike_times_cell			
	else:
		data_dict[out_str[it]]=out[it]

#Saving data to Matlab format
io.savemat(output_dir+'/Matlab_output_'+sub_run_name+'/data',data_dict)

# Stop timer:
tEnd = tim.mktime(tim.localtime())
comp_time=tim.strftime("H:%H M:%M S:%S",tim.gmtime(tEnd - tBegin))
print(comp_time)
