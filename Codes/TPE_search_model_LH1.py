# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:36:29 2025


"""

import numpy as np
import pandas as pd
import optuna
# import joblib
# from optuna.visualization.matplotlib import plot_optimization_history
from functools import partial
import matplotlib.pyplot as plt
from kinetic_and_regression_functions import Det_criterion,model_LH1,Model_simulation


# import intial conditions and experimental data

input_data  = pd.read_excel('synthetic_data.xlsx',sheet_name = 'Exp_data' , dtype=np.float64 )
initial_data = pd.read_excel('input_data.xlsx' ,sheet_name = 'init_cond', dtype=np.float64 ) # initial conditions data

initial_conditions = initial_data[["RUN","thermal_mode","m_FF_g","m_FA_g","m_2MF_g","m_H2O_g","Cat","Temp","p_H2"]].to_numpy()
exp_data = input_data[['RUN','thermal_mode','time (sec)','weight_f','CFF (mol/m3)','CFA (mol/m3)','C2MF (mol/m3)','Temp K']].to_numpy() # experimental data


# isothermal data 

# import parameters

Run_no = np.unique([1,2,3,4,7,11,15,19]) # specify the runs in isothermal mode that you want to use

X_input = [] # empty  list
Y_output = [] # empty list


# extract data for the specified Run_no
for i in sorted(Run_no):
    X_input.append(initial_conditions[initial_conditions[:,0]==i])
    Y_output.append(exp_data[exp_data[:,0] == i])



X_data = np.vstack(X_input) # experimental settings and initial conditions

Y_data = np.vstack(Y_output) # experimental data

m1 = Model_simulation(model_LH1) # model 1 object from model simulation class


Det_func= partial(Det_criterion,model_sim =m1.simulate,X_train =  X_data,
                  Y_train = Y_data)  # The in-built partial function allows to fix the arguments of a define function

# define the search speaces to be used with TPE
def space_1(trial1):
    
    par = [trial1.suggest_float('ln_k01',-10, 0),
           trial1.suggest_float('ln_k02',-10, 0),
           trial1.suggest_float('Ea_RT1',0, 50),
           trial1.suggest_float('Ea_RT2',0, 50),
           trial1.suggest_float('KFF',1, 10),
           trial1.suggest_float('KFA',1, 10),
           trial1.suggest_float('K2MF',1, 10),
           trial1.suggest_float('KH', 0, 100),
           trial1.suggest_float('KW',0, 10)]
       
    return Det_func(par)

def space_2(trial2):
    
    par = [trial2.suggest_float('ln_k01',-10, 0),
           trial2.suggest_float('ln_k02',-10, 0),
           trial2.suggest_float('Ea_RT1',0, 50),
           trial2.suggest_float('Ea_RT2',0, 50),
           trial2.suggest_float('KFF',1e-2, 1),
           trial2.suggest_float('KFA',1e-2, 1),
           trial2.suggest_float('K2MF',1e-2, 1),
           trial2.suggest_float('KH', 0, 100),
           trial2.suggest_float('KW',0, 10)]
       
    return Det_func(par)

# sampler, use the seed to obtain reproducible results

sampler = optuna.samplers.TPESampler(seed = 123)


#%% search space 1


study1   = optuna.create_study(sampler=sampler, direction = 'minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING) # limits the verbosity to wernings

study1.optimize(space_1, n_trials= 5000,  show_progress_bar = True, n_jobs= 1, timeout=3600, catch = (ValueError)) # -1 uses all the CPU CORES

search1_vals = study1.best_trial.params # best parameters
sampled_data1 = study1.trials_dataframe() # save the results in data frame


ax1 = optuna.visualization.matplotlib.plot_optimization_history(study1)

ax1.set_title('Search 1')
ax1.tick_params(axis='both', labelsize = 13, labelcolor = 'k')
ax1.set_xlabel( "Trial", fontsize = 13, color = 'black')
ax1.set_ylabel('Objective value 'r"$ln|v(\theta)|$", fontsize = 13, color = 'black', fontfamily = 'verdana')
line = ax1.get_lines()

for line in ax1.get_lines():
    line.set_linewidth(1.3)
    line.set_color('firebrick')


order = [0,1]
handles, labels = plt.gca().get_legend_handles_labels()
ax1.legend( handles=[handles[i] for i in order], labels=[labels[i] for i in order] ,loc='center', bbox_to_anchor=(0.5,-0.25), fontsize = 13,
            fancybox=True, shadow=False, ncol=2)
plt.savefig('search1_optimization_plot.png', dpi = 700, bbox_inches='tight' )


#%% search space 2 

study2   = optuna.create_study(sampler=sampler, direction = 'minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING) # limits the verbosity to warnings

study2.optimize(space_2, n_trials= 5000,  show_progress_bar = True, n_jobs= 1, timeout=3600, catch = (ValueError)) # -1 uses all the CPU CORES

search2_vals = study2.best_trial.params # best parameters found 
sampled_data2 = study2.trials_dataframe() # convert the results to dataframe


# plot the search history
ax2 = optuna.visualization.matplotlib.plot_optimization_history(study2)

ax2.set_title('Search 2')
ax2.tick_params(axis = 'both', labelsize = 13, labelcolor = 'k')
ax2.set_xlabel( "Trial", fontsize = 13, color = 'black')
ax2.set_ylabel('Objective value 'r"$ln|v(\theta)|$", fontsize = 13, color = 'black', fontfamily = 'verdana')
line = ax2.get_lines()

for line in ax2.get_lines():
    line.set_linewidth(1.3)
    line.set_color('firebrick')


order = [0,1]

handles, labels = plt.gca().get_legend_handles_labels()
ax2.legend( handles=[handles[i] for i in order], labels=[labels[i] for i in order] ,loc='center', bbox_to_anchor=(0.5,-0.25), fontsize = 13,
            fancybox=True, shadow=False, ncol=2)
plt.savefig('search2_optimization_plot.png', dpi = 700, bbox_inches='tight' )



#%% save files

# import joblib

# joblib.dump(study1, "search-1_TPE_5000_trials_8runs.pkl")

# joblib.dump(study2, "search-2_TPE_5000_trials_8runs.pkl")

sampled_data1.to_excel("TPE-8_runs_search1.xlsx") # results from study 1
sampled_data2.to_excel("TPE-8_runs_search2.xlsx") # results from study 2

#%% plot 


par = list(search1_vals.values()) # extract parameters values from the dictionary


for i in range(len(sorted(Run_no))):    
    
    rx_time = Y_output[i][:,2]/60 # in min
    
    # simulation data
    
    iso_sim,_= m1.simulate(X_input[i],Y_output[i], par)
    
    sim_data = iso_sim*1e-3 # mol/L

    
    fig, ax1 = plt.subplots()

    # concentration plots
    ax1.plot(rx_time,sim_data[:,0],color = 'firebrick', label = 'FF',alpha = 1, linewidth= 0.7)
    ax1.plot(rx_time,sim_data[:,1],color = 'royalblue', label = 'FA',alpha = 1, linewidth= 0.7)
    ax1.plot(rx_time,sim_data[:,2],color = 'blueviolet', label = '2MF',alpha = 1, linewidth= 0.7)
    
    ax1.scatter(rx_time,Y_output[i][:,4]*1e-3,color = 'firebrick', label = 'exp_FF',alpha = 1, marker = 'o', facecolors='none')
    ax1.scatter(rx_time,Y_output[i][:,5]*1e-3,color = 'royalblue', label = 'exp_FA',alpha = 1, marker = '^', facecolors='none')
    ax1.scatter(rx_time,Y_output[i][:,6]*1e-3,color = 'blueviolet', label = 'exp_2MF',alpha = 1, marker = 's', facecolors='none')
    
  
        
    
    ax1.set_xlabel('Time (min)', fontsize = 13)
    ax1.tick_params(axis='both', labelsize=13)  # Adjust the size as needed
    ax1.set_ylabel(' Concentration (mol/$L$)',fontsize = 13)
    
    
    order = [0,3,1,4,2,5]
    # ax1.legend(loc = 'center left')
    handles, labels = plt.gca().get_legend_handles_labels()
    ax1.legend( handles=[handles[i] for i in order], labels=[labels[i] for i in order] ,loc='center', bbox_to_anchor=(0.5,-0.30), fontsize = 13,
                fancybox=True, shadow=False, ncol=3)

    ax1.set_title("Run " + str(int(X_input[i][0,0])),fontsize = 13)  # Add a title to the axes.
    plt.show()
