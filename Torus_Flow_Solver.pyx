#JAN 2014 -- GUILLAUME LAJOIE	

#Cython compilable module to return the integration of ONE initial condition of a network of N coupled theta-neurons (on the N-torus). 

#Returns a choice of values: trajectory of all cells, Classical LE trace, FTLE trace and nbr of rotation of a subset of cells. 
#In addition, the code samples a histogram of trajectory points for the invariant measure as it goes along.

#To be used with the associated setup_TF.py file to compile 

#Notes : coupling matrix A={a_ij} represent coupling from neuron j to neuron i !!! j->i !!! 
#NO AUTAPSES ... Jacobian cannot deal with autapses so diagonal of A must be zero !!!!!!!!!!!!

#IMPORT NEEDED MODULES
import random, uuid
import numpy as np
cimport numpy as np
import math as ma
cimport cython
from sys import stdout

#COMPILE-TIME TYPE INITIALIZATION FOR NUMPY ARRAYS:
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTINT_t
# numpy array initializations:
DTYPE = np.float	# Initialize a data-type for the array	
DTINT = np.int	

# External math wrapper functions that might be needed:
cdef extern from "math.h":
	DTYPE_t sqrt(DTYPE_t sqrtMe)
	DTYPE_t cos(DTYPE_t sqrtMe)
	DTYPE_t sin(DTYPE_t sqrtMe)
	DTYPE_t pow(DTYPE_t sqrtMe, DTYPE_t dummy)
	DTYPE_t fmod(DTYPE_t a, DTYPE_t b)
	DTYPE_t fabs(DTYPE_t a)
	DTYPE_t log(DTYPE_t a)
	DTYPE_t exp(DTYPE_t a)

cdef DTYPE_t pi = np.pi

#++++++++++++++++++++++++++++++	
# Wrapper for the RNG:
cdef extern from "MersenneTwister.h":
	ctypedef struct c_MTRand "MTRand":
		DTYPE_t randNorm( double mean, double stddev)
		void seed( unsigned long bigSeed[])
#++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++
# RNG initializations:
cdef unsigned long mySeed[624]		# Seed array for the RNG, length 624
cdef c_MTRand myTwister				# RNG object construction
# Initialization of random number generator:
myUUID = uuid.uuid4()
random.seed(myUUID.int)
for i in range(624): mySeed[i] = random.randint(0,int(exp(21)))
myTwister.seed(mySeed)
#++++++++++++++++++++++++++++++++++++++++++++


#DEFINING INTERNAL FUNCTIONS+++++++++++++++++++++++++++++++++++++++++++++++++++++++


#DEFINING Graam-Schmidth function for LE spectrum computation
cdef np.ndarray[DTYPE_t, ndim=1] GS(np.ndarray[DTYPE_t, ndim=2] A):
	cdef DTINT_t k=np.shape(A)[1]
	cdef np.ndarray[DTYPE_t, ndim=1] v_norm=np.zeros(k)
	cdef Py_ssize_t it, jit
	for it in range(k):
		v_norm[it]=sqrt(np.dot(A[:,it],A[:,it]))
		A[:,it]=A[:,it]/v_norm[it]
		for jit in range(it+1,k):
			A[:,jit]=A[:,jit]-np.dot(A[:,jit],A[:,it])/np.dot(A[:,it],A[:,it])*A[:,it]
	return v_norm

#toral distance function
cdef DTYPE_t toral_dist(np.ndarray[DTYPE_t, ndim=1] v, np.ndarray[DTYPE_t, ndim=1] w, DTINT_t N):
	cdef DTYPE_t temp=0
	cdef Py_ssize_t it
	for it in range(N):
		temp=temp+pow(min(abs(v[it]-w[it]),1-abs(v[it]-w[it])),2)
	return np.sqrt(temp)
	

#g : -> bump function
cdef np.ndarray[DTYPE_t, ndim=1] g(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t b):
	cdef np.ndarray[DTINT_t, ndim=1] index
	cdef DTYPE_t c
	cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(len(x), dtype=DTYPE)
	
	c=35./32.*pow(b,-7)
	x = np.fmod(x+0.5,1.) - 0.5
	index=np.nonzero(np.fabs(x)<b)[0]
	y[index]=c * np.power(( pow(b,2) - np.power(x[index],2) ),3)
	return y
	
#g' : -> bump function derivative
cdef DTYPE_t g_prime(DTYPE_t x, DTYPE_t b):
	cdef DTYPE_t c
	cdef DTYPE_t y=0. 
	
	c=35./32.*pow(b,-7)
	x = fmod(x+0.5,1.) - 0.5
	if fabs(x)<b:
		y=-6.0*c * pow(( pow(b,2) - pow(x,2) ),2) * x
	return y    

#VF : -> vector field (undriven) ... IN ITO FORM
cdef np.ndarray[DTYPE_t, ndim=1] VF(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] eta, np.ndarray[DTYPE_t, ndim=2] A, DTYPE_t b, np.ndarray[DTYPE_t, ndim=1] epsilon):
	cdef np.ndarray[DTYPE_t, ndim=1] y=np.zeros(len(x),  dtype=DTYPE)
	y=1+np.cos(2.0*pi*x)+eta*(1-np.cos(2.0*pi*x))+np.dot(A,g(x,b))*(1-np.cos(2.0*pi*x))+pi*np.power(epsilon,2)*(1-np.cos(2.0*pi*x))*np.sin(2.*pi*x)
	return y
	
#Jacobian of VF : -> without stochastic part IN ITO FORM
cdef np.ndarray[DTYPE_t, ndim=2] Jacob(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] eta, np.ndarray[DTYPE_t, ndim=2] A, DTYPE_t b, np.ndarray[DTYPE_t, ndim=1] epsilon):
	cdef np.ndarray[DTYPE_t, ndim=2] J = np.ndarray([len(x),len(x)], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] couplage=np.dot(A,g(x,b))
	cdef Py_ssize_t it, jit
	for it in range(0,len(x)):
		J[it,it]=2.0*pi*sin(2.*pi*x[it])*(eta[it]-1.+couplage[it]+pi*pow(epsilon[it],2)*sin(2.*pi*x[it])*cos(2.*pi*x[it]))
		#J[it,it]=-2.0*pi*sin(2.0*pi*x[it])+(eta[it]+couplage[it])*2.0*pi*sin(2.0*pi*x[it]) <<-- old form
		for jit in list(range(0,it))+list(range(it+1,len(x))):
			J[it,jit]=(1-cos(2.0*pi*x[it]))*A[it,jit]*g_prime(x[jit],b) #as it was [it,jit]
	return J

#stoch_VF_filter : -> stochastic forcing filter
cdef np.ndarray[DTYPE_t, ndim=1] stoch_VF_filter(np.ndarray[DTYPE_t, ndim=1] x):
	cdef np.ndarray[DTYPE_t, ndim=1] y=np.zeros(len(x),  dtype=DTYPE)
	y=1-np.cos(x*2.*pi)              
	return y
	
#stochastic part of jacobian
cdef np.ndarray[DTYPE_t, ndim=2] stoch_Jacob(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] epsilon):
	cdef np.ndarray[DTYPE_t, ndim=2] J = np.ndarray([len(x),len(x)], dtype=DTYPE)
	J=np.diag(epsilon*np.sin(2.*pi*x))*2.*pi
	return J	

#MAIN FUNCTION OF MODULE++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def N_torus_integrate(
					DTINT_t N,				#number of cells
					DTINT_t N_rec,			#number of cells to record data from
					np.ndarray[DTINT_t, ndim=1] rec_index,	#index of cells to be recorded
					np.ndarray[DTINT_t, ndim=1] D_index, 	#index of driven cells
					np.ndarray[DTYPE_t, ndim=1] eta, 		#cells param N-vector !!!!!
					np.ndarray[DTYPE_t, ndim=2] A,			#coupling matrix NxN !!!!!
					np.ndarray[DTYPE_t, ndim=1] epsilon, 	#input signal strengths N-vector!!!!!!!!
					DTYPE_t b, 			#bump support
					DTYPE_t t_span, 	#integration time length
					DTYPE_t dt, 		#integration timestep
					DTYPE_t dt_sample, 	#lyapunov exp recording timestep
					DTINT_t sample_save_increment, #number of sampling computations (at each dt_sample) between saves
					DTINT_t hist_bin_num, #number of bins on [0,1] for histogram sampling of trajectories
					np.ndarray[DTYPE_t, ndim=1] x, #Initial condition N-vector!!!!!!!!!
					DTYPE_t c, #drive correlation parameter
					np.ndarray[DTINT_t, ndim=1] seedList,
					DTYPE_t xi, #perturbation for pseudo traj
					DTYPE_t pseudo_dt,
					DTYPE_t mu_noise, 
					DTINT_t k_LE_spectrum,
					DTINT_t traj_switch,
					DTINT_t spike_times_switch,
					DTINT_t LE_switch,
					DTINT_t FTLE_switch, #Decides what to save etc (0 or 1)... see list in header
					DTINT_t pseudo_LE_switch, #Decide to perfomr pseudo LE approx with secondary perturbed traj
					DTINT_t drive_save_switch, #Decides to save drive vector
					DTINT_t print_progress #Decides to print sim progress on terminal
					):
	
	#---------------------------------------------------------
	#Declaring variables needed by the solver
	#---------------------------------------------------------
	#NOTE: LE_buffer is the size of arrays that that sample trajectories and LE_traces
	#---------------------------------------------------------

	#GENERAL++++++++++++++++
	cdef DTINT_t timestep_counter=0, N_prep, counter = 0, LE_buffer=np.floor(t_span/(sample_save_increment*dt_sample))-1, dummy_int, N_D, pseudo_counter=0
	cdef DTYPE_t t_sample = dt_sample, t_record=sample_save_increment*dt_sample, t_now, norm, LE_sum = 0.
	cdef Py_ssize_t it, i
	cdef np.ndarray[DTINT_t, ndim=1] index
	cdef np.ndarray[DTINT_t, ndim=1] index_rec
	cdef np.ndarray[DTINT_t, ndim=1] rot=np.zeros(N, dtype=DTINT)
	cdef np.ndarray[DTYPE_t, ndim=1] vect_lyap=np.random.normal(0,1,N) #sampling a new vector
	cdef np.ndarray[DTYPE_t, ndim=2] drive_container = np.ndarray([N_rec,drive_save_switch*LE_buffer], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] save_times = np.ndarray(LE_buffer, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] x_temp = np.ndarray(N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] x_old = np.ndarray(N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] dx = np.ndarray(N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] delta_t = np.ndarray(N, dtype=DTYPE)
	cdef np.ndarray[DTINT_t, ndim=1] spike_counter=np.zeros(N_rec, dtype=DTINT)
	cdef np.ndarray[DTINT_t, ndim=2] hist_traj_matrix=np.zeros([N,hist_bin_num], dtype=DTINT)
	cdef np.ndarray[DTYPE_t, ndim=1] hist_bin_points=np.ndarray(hist_bin_num+1, dtype=DTYPE)
	hist_bin_points=np.linspace(0,1,hist_bin_num+1)
	cdef np.ndarray[DTYPE_t, ndim=1] drive=np.zeros(N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] drive_temp=np.zeros(len(D_index), dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] noise_temp=np.zeros(len(D_index), dtype=DTYPE)
	cdef np.ndarray[DTINT_t, ndim=1] dummy_index=np.zeros(N, dtype=DTINT)

	#TRAJECTORY CONTAINER++++++++++++
	cdef np.ndarray[DTYPE_t, ndim=2] traj_container = np.ndarray([traj_switch*N_rec,traj_switch*LE_buffer], dtype=DTYPE) 

	#SPIKE TIMEs CONTAINER++++++++++++
	#NOTE: Initiate empty container big enough to store 3 spikes per time-unit per cell. That is enough for most parameters
	#and small enough not to hog too much memory. This might have to be adjusted...
	cdef np.ndarray[DTYPE_t, ndim=2] spike_times_container=np.zeros([spike_times_switch*N_rec ,spike_times_switch*np.floor(2.*t_span)], dtype=DTYPE)

	#LE SPECTRUM CONTAINERS AND VARIABLES+++++++++++++LE_switch*
	cdef np.ndarray[DTYPE_t, ndim=2] J = np.ndarray([LE_switch*N,LE_switch*N], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=2] STM=np.eye(LE_switch*k_LE_spectrum,dtype=DTYPE) #standard transition matrix for FTLE computing
	STM=np.append(STM,np.zeros([LE_switch*(N-k_LE_spectrum),LE_switch*k_LE_spectrum], dtype=DTYPE),axis=0)
	cdef np.ndarray[DTYPE_t, ndim=1] full_norms = np.ndarray(LE_switch*k_LE_spectrum, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] full_LE_sum = np.zeros(LE_switch*k_LE_spectrum, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=2] full_LE_container = np.ndarray([LE_switch*k_LE_spectrum,LE_switch*LE_buffer], dtype=DTYPE)
	#Full Traj LE recordings (FTLE_switch) containers 
	cdef np.ndarray[DTYPE_t, ndim=2] full_LE_norms = np.ndarray([FTLE_switch*LE_switch*k_LE_spectrum,FTLE_switch*LE_switch*LE_buffer], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=3] full_LE_basis_container= np.ndarray([FTLE_switch*LE_switch*N,FTLE_switch*LE_switch*k_LE_spectrum,FTLE_switch*LE_switch*LE_buffer], dtype=DTYPE)

	#PSEUDO LE CONTAINERS AND VARIABLES
	cdef DTYPE_t pseudo_t=pseudo_dt
	cdef np.ndarray[DTYPE_t, ndim=2] pseudo_traj_container = np.ndarray([pseudo_LE_switch*N_rec,pseudo_LE_switch*LE_buffer], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] pseudo_x_temp = np.ndarray(pseudo_LE_switch*N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] pseudo_x_old = np.ndarray(pseudo_LE_switch*N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] pseudo_x = np.ndarray(pseudo_LE_switch*N, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] pseudo_ly= np.zeros(pseudo_LE_switch*np.floor(t_span/pseudo_dt)+1,dtype=DTYPE)
	
	#creating map ordered map for recordings indices
	for it in range(N):
		if it in rec_index:
			dummy_index[it]=counter
			counter =counter+1
	counter=0
	
	#Items fro RNG----------------------------------------------------
	cdef unsigned long mySeed[624]	# Seed array for the RNG, length 624
	cdef c_MTRand myTwister				# RNG object construction
	#-----------------------------------------------------------------
	
	#Initializing RNG using passed seed-------------------------------
	if seedList.all()!=0:
		for i in range(624): mySeed[i]=seedList[i]
		myTwister.seed(mySeed)
	#-----------------------------------------------------------------
	
	N_D=len(D_index) #number of driven cells
		
	norm=sqrt(np.dot(vect_lyap,vect_lyap))
	vect_lyap=vect_lyap/norm #renormalizing
	
	#pseuod LE perturbed traj initialization
	pseudo_x=x+xi*vect_lyap

		
	#START EULER INTEGRATION++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	t_now=dt
	while t_now < t_span: #time integration
		#print(t_now)
		timestep_counter+=1
		
		for it in range(N_D):
			drive_temp[it]=myTwister.randNorm(0,1)
		drive[np.array(D_index)]=sqrt(1.-c)*drive_temp+sqrt(c)*myTwister.randNorm(0,1) #generating white noise step

		if mu_noise!=0: #scaling in noise so that variance of stochastic part is still epsilon
			noise_temp=np.random.normal(0,1,N_D)
			drive[np.array(D_index)]=sqrt(1.-mu_noise)*drive[np.array(D_index)]+sqrt(mu_noise)*noise_temp

		
		
		x_old=x
		x_temp=x+VF(x, eta, A, b, epsilon)*dt #deterministic update
		x=x_temp+stoch_VF_filter(x)*sqrt(dt)*(drive*epsilon) #stoch drive step
		
		#pseudo traj evolve
		if pseudo_LE_switch==1:
			pseudo_x_old=pseudo_x
			pseudo_x_temp=pseudo_x+VF(pseudo_x, eta, A, b, epsilon)*dt #deterministic update
			pseudo_x=pseudo_x_temp+stoch_VF_filter(pseudo_x)*sqrt(dt)*(drive*epsilon) #stoch drive step
			pseudo_x=np.mod(pseudo_x,1.0)
		
		
		index=np.nonzero(x>1.0)[0]
		rot[index]=rot[index]+1 #updating rotation numbers
		
		if spike_times_switch==1:
			index_rec=np.array(list(set(index) & set(rec_index)),dtype=int) 
			dx=x-x_old
			delta_t=(x-1)*dt/dx
			spike_times_container[dummy_index[index_rec],spike_counter[dummy_index[index_rec]]]=t_now-delta_t[index_rec]
			spike_counter[dummy_index[index_rec]]+=1
		x=np.mod(x,1.0)
		    
	   #COMPUTING LYAPUNOV EXPANSION QUANTITIES------------------------------------------
	   #Deterministic and stochastic parts of jacobian
		if LE_switch==1: 
			J=Jacob(x_old,eta,A,b,epsilon)*dt+stoch_Jacob(x_old,epsilon)*drive*sqrt(dt)
			
		#Numerical integration step to evolve STM (standard transition matrix) for FTLE
		if LE_switch==1:
			STM=STM+np.dot(J,STM)
	 
				
		t_now += dt # update time
		#print dt_adapt
		
		#renormalizing pseudo trajectory in direction of difference with base traj
		if pseudo_LE_switch==1:
			if t_now > pseudo_t:
				pseudo_ly[pseudo_counter]=log(toral_dist(pseudo_x,x,N)/xi)/pseudo_dt
				pseudo_counter=pseudo_counter+1
				pseudo_x_temp=pseudo_x-x
				norm=sqrt(np.dot(pseudo_x_temp,pseudo_x_temp))
				pseudo_x_temp=pseudo_x_temp/norm
				pseudo_x=x+xi*pseudo_x_temp
				pseudo_x=np.mod(pseudo_x,1.0)
				pseudo_t=pseudo_t+pseudo_dt	
		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
		#Rescale LE matrix STM------------------------------------------------ 
		if t_now > t_sample:

			if LE_switch==1:
				full_norms=GS(STM)
				full_LE_sum=full_LE_sum+np.log(full_norms)

			#Saving to containers++++++++++++++++++++++++++++++++++++++++++++++++++++
			if t_now >= t_record and counter < LE_buffer:

				#printing sim progress
				if print_progress==1:
					stdout.write('                                                ')
					stdout.write('\r')
					stdout.flush()
					stdout.write('Simulation progress: '+str(round(100*(t_record/t_span)))+' %')
					stdout.write('\r')
					stdout.flush()

				save_times[counter]=t_now
				if drive_save_switch==1:
					drive_container[:,counter]=drive[rec_index]
				
				for it in range(N):					
					dummy_int=np.searchsorted(hist_bin_points,x[it])
					hist_traj_matrix[it,dummy_int-1] += 1
				
				if traj_switch==1:
					traj_container[:,counter]=x[rec_index]
					if pseudo_LE_switch==1:
						pseudo_traj_container[:,counter]=pseudo_x[rec_index]
					
				if LE_switch==1:
					full_LE_container[:,counter]=full_LE_sum/t_now
					if FTLE_switch==1:
						full_LE_basis_container[:,:,counter]=STM
						full_LE_norms[:,counter]=full_norms
					
				counter=counter+1
				t_record=t_record+sample_save_increment*dt_sample #updating next save time
					   
			t_sample=t_sample+dt_sample #updating next sample time
				

	#END EULER INTEGRATION+++++++++++++++++++++++++++++++++++++++++++++ 
	stdout.write('\n')
	stdout.flush()  
	print('run is done')
	print('final time: '+str(round(t_now)))
	
	#RETURN STATEMENTS DEPENDING ON SWITCHES+++++++++++++++++++++++++++
	
	#common to all outputs
	out_str=['save_times','rot','hist_traj_matrix']
	out=(save_times,rot,hist_traj_matrix)
	
	#running trhough switchboard
	if LE_switch==1:
		out_str.append('full_LE_container')
		out=out+(full_LE_container,)
		if FTLE_switch==1:		
			out_str.append('full_LE_basis_container')
			out=out+(full_LE_basis_container,)
			out_str.append('full_LE_norms')
			out=out+(full_LE_norms,)

	if traj_switch==1:
		out_str.append('traj_container')
		out=out+(traj_container,)
		
	if spike_times_switch==1:
		out_str.append('spike_times_container')
		out=out+(spike_times_container,)				
		
	if pseudo_LE_switch==1:
		out_str.append('pseudo_ly')
		out=out+(pseudo_ly,)
		if traj_switch==1:
			out_str.append('pseudo_traj_container')
			out=out+(pseudo_traj_container,)
		
	if drive_save_switch==1:
		out_str.append('drive_container')
		out=out+(drive_container,)
	
	#output		
	return out_str, out


#END MAIN FUNCTION OF MODULE++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    
 

