# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:03:43 2024

@author: Erny Encarnacion Munoz
"""
import numpy as np
import pandas as pd
# from scipy.optimize import minimize,least_squares
import optuna
from scipy import stats
from scipy.stats import probplot
# from scipy.stats import norm
import numdifftools as nd
from functools import partial
import matplotlib.pyplot as plt
from kinetic_and_regression_functions import Det_criterion,model_LH1,Statistical_inferece,Model_simulation,split_data



#%% importing data
# import intial conditions and experimental data

# import intial conditions and experimental data
input_data  = pd.read_excel('synthetic_data.xlsx',sheet_name = 'Exp_data' , dtype=np.float64 )
initial_data = pd.read_excel('input_data.xlsx' ,sheet_name = 'init_cond', dtype=np.float64 ) # initial conditions data



initial_conditions = initial_data[["RUN","thermal_mode","m_FF_g","m_FA_g","m_2MF_g","m_H2O_g","Cat","Temp","p_H2"]].to_numpy()
exp_data = input_data[['RUN','thermal_mode','time (sec)','weight_f','CFF (mol/m3)','CFA (mol/m3)','C2MF (mol/m3)','Temp K']].to_numpy()     # experimental data

# import parameters
par_model_1 = pd.read_excel('input_data.xlsx', sheet_name ='parameters_M1') # model1
par_labels = par_model_1['parameters'].tolist() # labels of parameters string

# parameters values
par1_0 = par_model_1['initial_values'].to_numpy()      #initial values model 1


# bounds of the parameters 
upbnd = par_model_1['uppbound'].tolist()
lwbnd = par_model_1['lowbound'].tolist()


bounds = tuple(par_model_1.loc[:,'lowbound':'uppbound'].to_numpy())




#%%% extract the data 

Run_no = np.unique([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # specify the runs in isothermal mode that you want to use



X_input = [] # empty  list
Y_output = [] # empty list

# extract data for the specified Run_no
for i in sorted(Run_no):
    X_input.append(initial_conditions[initial_conditions[:,0]==i])
    Y_output.append(exp_data[exp_data[:,0] == i])



X_data = np.vstack(X_input) # experimental settings and initial conditions

Y_data = np.vstack(Y_output) # experimental data

m1 = Model_simulation(model_LH1) # model 1 object from model simulation class


#%% optimization space


def space_1(trial1):
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)    #limits the verbosity to warnings
    
    par = [trial1.suggest_float(par_labels[i],lwbnd[i],upbnd[i]) for i in range(len(par_labels))] # list of parameters suggested by the sampler
    
    
    # if any([trial1.number == 1000 ]) == True   : # you can specify when to use Gradient optimization method
         
    #     # par_opt = par1_0
    #     print('\n ==== Gradient based optmization in process =====\n')
    #     par_trial = np.array(list(study1.best_trial.params.values())) # best parameter trial
        
    #     par_opt = least_squares(SSE_func, x0 = par_trial, method = 'trf',bounds = (lwbnd,upbnd ),\
    #                                 args = (m1.simulate,X_data,Y_data)) # optimal parameters
        
    #     # par_opt = minimize(Det_criterion, x0 = par_trial,  method ='L-BFGS-B',options = {'maxiter':2000,  'ftol': 1e-5, 'gtol': 1e-5}, \
    #     #                                               bounds = bounds, args = (m1.simulate,X_data,Y_data))    
        
    #     # par_opt = minimize(Det_criterion, x0 = par_trial,  method ='Nelder-Mead',options = {'maxiter':7000, 'adaptive': True,  'fatol': 1e-7, 'xatol': 1e-7}, \
    #     #                                               bounds = bounds, args = (m1.simulate,X_data,Y_data))        
    #     par_gdb = par_opt.x    
    #     par_lsq = []
    #     for i, j in zip(par_labels,par_gdb):
    #         par_lsq.append((i,j))
    #     par_lsq =  dict(par_lsq) 
        
        
    #     study1.enqueue_trial(par_lsq, user_attrs= {'method': 'trusted region reflective', 'objective': 'sum of squared error'})
    #     print(' optimal parameters :\n ', par_lsq, 'function value = {:.4f}'. format(Det_func(par_gdb)) ,'\n' )
     
    
    return Det_func(par)


# # #%% parameter estimation



Det_func = partial(Det_criterion,model_sim = m1.simulate, X_train = X_data,
                    Y_train = Y_data) # The in-built partial function allows to fix the arguments of a define function




# first use TPE to explore the parameter space 

#%%% TPE sampling
sampler = optuna.samplers.TPESampler(seed = 123)

study1  = optuna.create_study(sampler = sampler, direction = 'minimize')



study1.optimize(space_1, n_trials = 1000, show_progress_bar = True, n_jobs= 1, timeout = 3600, catch = (ValueError)) # -1 uses all the CPU CORES

search1_best = study1.best_trial.params # dictionary of parameters found by TPE




#%%% CMA-ES sampling


study1.sampler =  optuna.samplers.CmaEsSampler(x0 = search1_best, seed = 13, restart_strategy= 'ipop', inc_popsize= 3) #use the initial values found by TPE
 

study1.optimize(space_1, n_trials = 6000, show_progress_bar = True , timeout=1.5*3600, catch = (ValueError)) # timeout in sec



sampled_data1 = study1.trials_dataframe() # save the results in data frame

#%% Inference regions
par_trial = np.array(list(study1.best_trial.params.values()))
# par = par1_0
par = par_trial # use the parameters found by CMA-ES algorithm

st = Statistical_inferece() # object to call the required functions from the statistical inference class

def iso_sim(parameters,model_sim,X_data,Y_data): 
     
      """ executes the model to simulate isothermal experiments"""
    
      sim_isothermal, _ = model_sim(X_data,Y_data,parameters)
         
      return sim_isothermal
    

res_iso = split_data(Y_data,Y_data)[0][:,[4,5,6]] - iso_sim(par,m1.simulate,X_data,Y_data) # residuals of isotohermal concentrations
err_var = np.dot(res_iso.T,res_iso)/(len(Y_data)-len(par)) # errors variance estimation
t_hpd = stats.t.ppf(1-0.05/2, (len(Y_data)-len(par)))      # t student value for DOF = N-P



#%%% HPD 

# computation of the Hessian matrix using finite difference of the ln |v(theta)|
Hess_matrix = nd.Hessian(Det_criterion, step = np.float64(1e-7))(par,m1.simulate,X_data,Y_data) 


pcov_hpd = 2*np.linalg.pinv(Hess_matrix) # parameter variances-covariance
    
hpd_95 =  t_hpd*np.sqrt(np.diag(pcov_hpd)) # HPD interval

cond_num_hpd = np.linalg.cond(pcov_hpd)    # condition number of the hessian

#%%% Correlation matrix of the FIM



grad = nd.Jacobian(iso_sim, step = np.float64(1e-10))(par,m1.simulate, X_data, Y_data) # jacobian

# loop to calculate te cov matrix at each u observation

FIM_t = [] # empty list for FIM calculation

for i in range(len(grad)):
    sens_m =grad[i,:,:] 
    FIM_i = (sens_m)@np.linalg.pinv(err_var)@sens_m.T # estimates covariance matrix using a linear approximation of the FIM
    FIM_t.append(FIM_i )

FIM_t = np.asarray(FIM_t) # total sensitivity matrix

cov_matrix = np.linalg.pinv(np.sum(FIM_t,axis =0)) # penroose inverse of the FIM

t_ci = stats.t.ppf(1-0.05/2, (len(Y_data)-len(par))) # t student value for DOF = N-P

ci_95 = t_ci*np.sqrt(np.diag(cov_matrix))

cond_num_ci = np.linalg.cond(cov_matrix)


CorMat_ci=np.zeros([len(cov_matrix),len(cov_matrix)])    
for j in range(len(cov_matrix)):
    for k in range(len(cov_matrix)):
          CorMat_ci[j,k]=cov_matrix[j,k]/(cov_matrix[j,j]*cov_matrix[k,k])**0.5




#%%plot data
# normal prop plot
res = st.res_vector(par,m1.simulate,X_data,Y_data) # residuals
fig,ax = plt.subplots( layout = 'tight')

fig.suptitle('Normal probability plot')
probplot(res.ravel(order='F'),\
        dist="norm", plot = ax,fit=True,rvalue=True)
ax.set_ylabel("Ordered residuals")
ax.set_title('regression using xx runs with model LH1')
plt.savefig('prob_plot_model-LH1 runs', dpi = 700, bbox_inches='tight' )

# par= par_trial
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
    ax1.set_title("Run " + str(int(i)))  # Add a title to the axes.
    
    
    order = [0,3,1,4,2,5]
    
    # ax1.legend(loc = 'center left')
    handles, labels = plt.gca().get_legend_handles_labels()
    ax1.legend( handles=[handles[i] for i in order], labels=[labels[i] for i in order] ,loc='center', bbox_to_anchor=(0.5,-0.30), fontsize = 13,
                fancybox=True, shadow=False, ncol=3)
    # plt.close()

    ax1.set_title("Run " + str(int(X_input[i][0,0])),fontsize = 13)  # Add a title to the axes.
    plt.show()
#%%% optimization plot
ax2 = optuna.visualization.matplotlib.plot_optimization_history(study1)

ax2.set_title('Optimization history with xx runs')
ax2.tick_params(axis='both', labelsize = 13, labelcolor = 'k')
ax2.set_xlabel( "Trial", fontsize = 13, color = 'black')
ax2.set_ylabel('Objective value 'r"$ln|v(\theta)|$", fontsize = 13, color = 'black', fontfamily = 'verdana')
line = ax1.get_lines()

for line in ax2.get_lines():
    line.set_linewidth(1.3)
    line.set_color('firebrick')


order = [0,1]
handles, labels = plt.gca().get_legend_handles_labels()
ax2.legend( handles=[handles[i] for i in order], labels=[labels[i] for i in order] ,loc='center', bbox_to_anchor=(0.5,-0.25), fontsize = 13,
            fancybox=True, shadow=False, ncol=2)
plt.savefig('xx_runs_optimization_plot.png', dpi = 700, bbox_inches='tight' )
#%%save results

cond_ci = pd.DataFrame(data = [cond_num_ci])
cond_hpd = pd.DataFrame(data =[cond_num_hpd])
confint_95 = pd.DataFrame(data = ci_95) 
hpd_95 = pd.DataFrame(data = hpd_95) 
sqres = pd.DataFrame(data = res**2)         
CorMat_ci = pd.DataFrame(data =CorMat_ci)
cov_ci = pd.DataFrame( data =cov_matrix )
cov_hpd = pd.DataFrame( data = pcov_hpd )
param = pd.DataFrame(data = par, index = par_labels)

init_cond = pd.DataFrame(data = X_data, columns=["RUN"	,"thermal_mode","m_FF"	,"m_FA","m_2MF","m_H2O"	,"Cat"	,"Temp","p_H2"])
regression_data = pd.DataFrame(data = Y_data, columns=['RUN','thermal_mode','time (sec)','weight_f','CFF (mol/m3)','CFA (mol/m3)','C2MF (mol/m3)','Temp K'])


with pd.ExcelWriter('M1-isothermal-RG_TPE-CMA_xx-runs.xlsx') as writer: # write data in a excel file
    param.to_excel(writer, sheet_name = 'optimal parameters')
    sqres.to_excel(writer, sheet_name = 'squared_residuals')
    CorMat_ci.to_excel(writer, sheet_name = 'corr mat_FIM' ) 
    cov_ci.to_excel(writer, sheet_name='Cov matrix_FIM')
    cond_ci.to_excel(writer,sheet_name='condnum_FIM')
    cov_hpd.to_excel(writer, sheet_name='Cov matrix_HPD')
    cond_hpd.to_excel(writer,sheet_name= 'condnum_HPD')
    hpd_95.to_excel(writer,sheet_name= '95% HPD')
    confint_95.to_excel(writer,sheet_name='95% CI')
    init_cond.to_excel(writer,sheet_name='initial conditions')
    regression_data.to_excel(writer, sheet_name='regression_exp_data')




sampled_data1.to_excel("M1-rg_results_xx-runs-TPE-CMA.xlsx") # results from study 1










