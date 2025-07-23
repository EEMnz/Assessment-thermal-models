# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:03:43 2024

@author: Erny Encarnacion Munoz
"""
import numpy as np
import pandas as pd
import numdifftools as nd
from kinetic_and_regression_functions import model_LH1,Statistical_inferece,Model_simulation,split_data
# import random
# import os
#%% importing data
# import intial conditions and experimental dataa

input_data  = pd.read_excel('synthetic_data.xlsx',sheet_name = 'Exp_data' , dtype=np.float64 )
initial_data = pd.read_excel('input_data.xlsx' ,sheet_name = 'init_cond', dtype=np.float64 ) # initial conditions data



initial_conditions = initial_data[["RUN","thermal_mode","m_FF_g","m_FA_g","m_2MF_g","m_H2O_g","Cat","Temp","p_H2"]].to_numpy()
exp_data = input_data[['RUN','thermal_mode','time (sec)','weight_f','CFF (mol/m3)','CFA (mol/m3)','C2MF (mol/m3)','Temp K']].to_numpy()     # experimental data

# isothermal data 

# import parameters
par_model_1 = pd.read_excel('input_data.xlsx', sheet_name ='M1_MBDOE_init') # model1

# parameters values
par1_0 = par_model_1['initial_values'].tolist()      #initial values model 1


par = par1_0



#%% initial Nq experiments 

Run_nq = np.unique([1, 2, 3, 4]) # initial available Nq experiments runs
X_nq = [] # empty  list
Y_nq = [] # empty list

# extract data for the specified Run_no
for i in sorted(Run_nq):
    X_nq.append(initial_conditions[initial_conditions[:,0]==i])
    Y_nq.append(exp_data[exp_data[:,0] == i])



X_data0 = np.vstack(X_nq) # experimental settings Nq and initial conditions
Y_data0 = np.vstack(Y_nq) # experimental data of available Nq experiments

# Nq + new experimets  to be added 
Run_new = np.unique([1, 2, 3, 4, 20])


X_new = [] # empty  list
Y_new = [] # empty list

# extract data for the specified Run_no
for i in sorted(Run_new):
    X_new.append(initial_conditions[initial_conditions[:,0]==i])
    Y_new.append(exp_data[exp_data[:,0] == i])



X_data_new = np.vstack(X_new) # experimental settings and initial conditions

Y_data_new = np.vstack(Y_new) # experimental dat

#%% model objects
m1 = Model_simulation(model_LH1) # model 1 object ffrom model simulation class

st = Statistical_inferece() # object to call the required functions from the statistical inference class


# %%Desing of experiments


"""  this desing is for multiresponse models in which the response variance covariance matrix is unknown

based on the article of : Box, M. J., and N. R. Draper. “Estimation and Design Criteria for Multiresponse Non-Linear 
                        Models with Non-Homogeneous Variance.” Journal of the Royal Statistical Society.
                        Series C (Applied Statistics), vol. 21, no. 1, 1972, pp. 13–24. JSTOR,
                        https://doi.org/10.2307/2346599. Accessed 17 Nov. 2024.

"""
def iso_sim(parameters,model_sim,X_data,Y_data): 
     
      """ executes the model to simulate isothermal experiments"""
    
      sim_isothermal, _ = model_sim(X_data,Y_data,parameters)
         
      return sim_isothermal

res_iso0 = split_data(Y_data0,Y_data0)[0][:,[4,5,6]] - iso_sim(par,m1.simulate,X_data0,Y_data0) # residuals of the responses


Nq = len(Run_nq) # number of available  experiments

Vq = np.dot(res_iso0.T,res_iso0) # residual matrix of Nq observations


X = nd.Jacobian(iso_sim, step = np.float64(1e-08))(par,m1.simulate, X_data_new, Y_data_new) #  jac of responses respect to par in each u observation

# X1 = X[:,:,0] # FF (Nxp) number of observations x parameters
# X2 = X[:,:,1] # FA (Nxp)
# X3 = X[:,:,2] # FA (Nxp)

m = 3 # number of responses

A_q = []
Vq_inv = np.linalg.pinv(Vq) # inverse 
for i in range(m):
    
    for j in range(m):
     
      A_q.append(Vq_inv[i,j]*(X[:,:,i].T@X[:,:,j])) # eq 5.14

   
A_t =  np.linalg.pinv(np.sum(Nq*A_q, axis = 0))     # eq.5.16

Doe_crit = np.linalg.slogdet(A_t)[1] # minimize the  determinant of the Cov matrix or FIM^-1

print('Determinant value =', Doe_crit)

