# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:24:59 2024

@author: erny_
"""

import numpy as np
import pandas as pd
import random
from kinetic_and_regression_functions import SSE_func,Det_criterion,Model_simulation,model_LH1,model_NCLH2,model_ELR3
from stratified_CV_functions import cross_val



#%%importin data 
# import intial conditions and experimental data

# import intial conditions and experimental data
input_data  = pd.read_excel('synthetic_data.xlsx',sheet_name = 'Exp_data' , dtype=np.float64 )
initial_data = pd.read_excel('input_data.xlsx' ,sheet_name = 'init_cond', dtype=np.float64 ) # initial conditions data



initial_conditions = initial_data[["RUN","thermal_mode","m_FF_g","m_FA_g","m_2MF_g","m_H2O_g","Cat","Temp","p_H2"]].to_numpy()
exp_data = input_data[['RUN','thermal_mode','time (sec)','weight_f','CFF (mol/m3)','CFA (mol/m3)','C2MF (mol/m3)','Temp K']].to_numpy()     # experimental data


# import parameters

# parameters
par_model_1 = pd.read_excel('input_data.xlsx', sheet_name ='parameters_M1') # model1
par_model_2 =pd.read_excel('input_data.xlsx', sheet_name ='parameters_M2') # model 2
par_model_3 =pd.read_excel('input_data.xlsx', sheet_name ='parameters_M3') # model 3


par_labels_1 = par_model_1['parameters'].tolist() # labels of parameters string
par_labels_2 = par_model_2['parameters'].tolist() # labels of parameters string
par_labels_3 = par_model_3['parameters'].tolist() # labels of parameters string

par0_1 =par_model_1['initial_values'].tolist()      #initial values
par0_2 =par_model_2['initial_values'].tolist()      #initial values
par0_3 =par_model_3['initial_values'].tolist()      #initial values

# #bounds
upp1 = par_model_1['uppbound'].tolist()         # upper bounds
low1 = par_model_1['lowbound'].tolist()         # lower bounds

upp2 = par_model_2['uppbound'].tolist()         # upper bounds
low2 = par_model_2['lowbound'].tolist()         # lower bounds

upp3 = par_model_3['uppbound'].tolist()         # upper bounds
low3 = par_model_3['lowbound'].tolist()         # lower bounds

#%% model objects
model1 = Model_simulation(model_LH1).simulate # model 1 object ffrom model simulation class
model2 = Model_simulation(model_NCLH2).simulate # model 2 object ffrom model simulation class
model3 = Model_simulation(model_ELR3).simulate # model 3 object ffrom model simulation class

obj_func = Det_criterion,SSE_func # objective functions



#%% 70/ 30 cross validation iso/ad

random.seed(156)
Run_isoth = sorted(random.sample(range(1,21), 15))
Run_adiabatic = sorted(random.sample(range(21,41),5))



Run_labels = np.concatenate((np.zeros(15),np.ones(5))) # labes of the runs

M3_CV_70 = cross_val("model_ELR3",par_labels_3,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model3,obj_func,par0_3,low3,upp3)

M2_CV_70 = cross_val("model_NCLH2",par_labels_2,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model2,obj_func ,par0_2,low2,upp2)

M1_CV_70 = cross_val("model_LH1",par_labels_1,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model1,obj_func ,par0_1,low1,upp1)


    


#%% 50/ 50 cross validation

random.seed(156)
# the number of sample generated has to match the classes generated
Run_isoth = sorted(random.sample(range(1,21), 10))
Run_adiabatic = sorted(random.sample(range(21,41),10))



Run_labels = np.concatenate((np.zeros(10),np.ones(10)))  # labes of the runs

M3_CV_50 = cross_val("model_ELR3",par_labels_3,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model3,obj_func,par0_3,low3,upp3)

M2_CV_50 = cross_val("model_NCLH2",par_labels_2,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model2,obj_func ,par0_2,low2,upp2)

M1_CV_50 = cross_val("model_LH1",par_labels_1,5,initial_conditions,exp_data,Run_isoth,Run_adiabatic,Run_labels,\
              model1,obj_func ,par0_1,low1,upp1)
