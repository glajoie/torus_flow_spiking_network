#GL JAN 2014

#Python module containing the main function "arch_maker" that creates
#a dictionary containing all parameters and data needed by the TF (torus flow)
#code

#Pckages needed
import numpy as np
import pickle as pkl
import random, uuid
from math import exp

def arch_maker(N,N_E,N_D,K,N_rec,num_epsilon,epsilon_min,epsilon_max,num_eta,eta_min,eta_max,num_c,c_min,c_max,num_IC,mu_noise,A_dial,i_eta_offset,eta_sample_width,b):

	#======================================
	#Making sure next parameters are of the good type
	#======================================
	N=int(N) #number of cells in network
	N_E=int(N_E) #number of excitatory cells in the network
	N_D=int(N_D) #number of driven cells
	N_rec=int(N_rec)
	N_I=N-N_E #number of inhibitory cells

	#======================================
	#Setting indices of cells to record from
	#======================================
	#NOTE: this will record from the first N_rec cells
	rec_index=np.array(range(N_rec))

	#======================================
	#Creating IC array for all subruns
	#======================================
	#Uniform distribution on the N-torus
	IC=np.random.uniform(0,1,[num_IC,N])

	#======================================
	#Generating random seed for drive
	#======================================
	myUUID = uuid.uuid4()
	fix_seed=myUUID.int
	random.seed(fix_seed)
	seedList=np.ndarray(624, dtype=int) #int--> long
	for i in range(624):
		seedList[i]=int(random.randint(0,int(exp(21)))) #first int -->long 
	 
	#======================================
	#Making parameter vectors (epsilon, eta and c)
	#======================================
	c_vector=np.linspace(c_min,c_max,num_c)
	epsilon_dial_vector=np.linspace(epsilon_min,epsilon_max,num_epsilon)
	eta_dial_vector=np.linspace(eta_min,eta_max,num_eta)

	#======================================
	#Assing E/I and drive identity to all cells (randomly)
	#======================================
	#generating indices of inhib/exit neurons
	s=np.random.permutation(range(N)) 
	I_index=s[0:N_I]
	E_index=s[N_I:N]
	#generating indices for driven cells
	D_index=np.random.permutation(range(N))[0:N_D]

	#======================================
	#Making network-wide cell parameters using
	#above param vectors as base of arrays.
	#======================================
	#building a list of rescaled eta vectors
	eta_E=np.random.uniform(0-eta_sample_width/2,0+eta_sample_width/2,N_E) 
	eta_I=np.random.uniform(0-eta_sample_width/2+i_eta_offset,0+eta_sample_width/2+i_eta_offset,N_I)
	eta_temp=np.zeros(N)
	for it in range(N_I):
		eta_temp[I_index[it]]=eta_I[it]
	for it in range(N_E):
		eta_temp[E_index[it]]=eta_E[it]
	eta=[]
	for dial in eta_dial_vector:
		eta.append(eta_temp+dial)
	#building a list of epsilon vectors
	epsilon=np.zeros(N)
	epsilon[np.array(D_index)]=1
	epsilon_temp=epsilon
	epsilon=[]
	for dial in epsilon_dial_vector:
		epsilon.append(dial*epsilon_temp) 

	#======================================
	#Making connectivity matrix 
	#======================================
	#NOTE: a_xy is connection strength from y to x
	a_ee=1./np.sqrt(K)
	a_ie=1./np.sqrt(K)
	a_ii=-.75/np.sqrt(K) #little offsetted as in VV and Somp. 96
	a_ei=-1./np.sqrt(K)
	#connectivity probabilities
	p_E=K/float(N_E)
	p_I=K/float(N_I)
	#making connectivity matrix
	A=np.zeros([N,N])
	for it in range(N):
		
		for jit in range(N):
			if it != jit:
				if jit in I_index:
					item=np.random.binomial(1,p_I) #generating random connectivity
					if it in E_index:
						A[it,jit]=a_ei*item
					else:
						A[it,jit]=a_ii*item
				else:
					item=np.random.binomial(1,p_E) #generating random connectivity
					if it in E_index:
						A[it,jit]=a_ee*item
					else:
						A[it,jit]=a_ie*item
	#rescaling A
	A=A_dial*A
	#perturbing A (multiplicative perturbation)
	A_perturb=np.random.normal(1,0.1,np.shape(A))
	A=A*A_perturb
		
	#======================================
	#Creating dictionary with architechture
	#======================================
	architecture={'N': N,'N_rec':N_rec, 'N_E':N_E, 'N_I':N_I,
		 'N_D':N_D, 'K':K, 'I_index':I_index, 'E_index':E_index, 'D_index':D_index, 'eta_E': eta_E,
		  'eta_I': eta_I, 'eta':eta, 'A': A, 'epsilon': epsilon, 'IC':IC, 'num_IC':num_IC,
		    'eta_dial_vector':eta_dial_vector, 'num_epsilon':num_epsilon,
		    'num_eta':num_eta, 'c_vector':c_vector, 'num_c':num_c, 'mu_noise':mu_noise,'epsilon_dial_vector':epsilon_dial_vector,
		    'seedList':seedList, 'rec_index':rec_index,'A_dial':A_dial,'i_eta_offset':i_eta_offset,'eta_sample_width':eta_sample_width,'b':b}

	#======================================
	#Return architecture dictionary
	#======================================
	return architecture
