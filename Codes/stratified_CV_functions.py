# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:52:01 2024

@author: Erny Encarnacion
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize,least_squares 
from sklearn.model_selection import StratifiedKFold
from functools import partial
import optuna

def split_data(Y_train,Y_test): # data input has to be in numpy

    # training data
    #================================================================================================
    #isothermal
    train_index_iso = np.argwhere(np.vstack(Y_train)[:,1]== 0) # extract index of isothermal data
    Y_train_iso = np.take(np.vstack(Y_train),train_index_iso.T[0],axis =0) # isothermal data for training
    
    # adiabatic data
    train_index_ad = np.argwhere(np.vstack(Y_train)[:,1]== 1) # extract index of adiabatic data
    Y_train_ad = np.take(np.vstack(Y_train),train_index_ad.T[0],axis =0) # isothermal data for training
    
    
    # testing data
    #================================================================================================
    #isothermal
    test_index_iso = np.argwhere(np.vstack(Y_test)[:,1]== 0) # extract index of isothermal data
    Y_test_iso = np.take(np.vstack(Y_test),test_index_iso.T[0],axis =0) # isothermal data for testing
    
    # adiabatic data
    test_index_ad = np.argwhere(np.vstack(Y_test)[:,1]== 1) # extract index of adiabatic data
    Y_test_ad = np.take(np.vstack(Y_test),test_index_ad.T[0],axis =0) # isothermal data for testing
    
    return Y_train_iso,Y_test_iso,Y_train_ad,Y_test_ad   

#%% cross validation

def cross_val(model_label,par_labels,splits_no,X_input,Y_output,Runs_iso,Runs_ad,Runs_class,\
              model_sim,obj_func,par_init,lowbnd,uppbnd):
    
    
    """
    # model label (str) : save and print the model used
    # Obj_func (func) : the objectives function, SSE function, determinant
    # model_sim (func): is the model function to simulate the data
    # splits_no : the number of splits to do the Cross Validation
    # Runs_class : list of 0 for isothermal and 1 for adiabatic, to clasify the runs
    # runs_iso : list of isothermal runs
    # runs_ad : list of adiabtic runs
    # X_input (D-array) : initial condiions 
    # Y_outpot (D-array) : experimentl data
    # par_labels (list) : string of parameters
    # par_init (init) : initial parameters
    # lowbnd (list): lower bounds of parameters
    # uppbnd(list) : upper bounds of parameters
    """
    
    
    # this loops converts the list of upper and lower bound to tupple 
    bounds = []
    for i, j in zip(lowbnd, uppbnd):
        bounds.append((i,j))
    
    bounds = tuple(bounds)  

    # variables declaration
    Run_labels        = Runs_class         # classes for the stratified CV
    par0              = par_init           # initial parameter values
    Run_isothermal    = Runs_iso           # isothermal Runs
    Run_adiabatic     = Runs_ad            # adiabatic Runs
    exp_data = Y_output                    # experimental data
   

    # fraction of isothermal and adiabatic experiments
    total_runs = len(Run_isothermal)+len(Run_adiabatic)
    frac_iso   = len(Run_isothermal)/total_runs
    frac_ad    = len(Run_adiabatic)/total_runs
    

    # class to fit and predict the data
    class kinetic_model:
        def __init__(self,objective_funcion,kinetic_model):
            
            # declare and initialize the variables to be used in the class
            self.model     = kinetic_model
            self.obj_func  = objective_funcion
            # self.fit_param = 0 # initialization
        pass
    
        
        def TPE_opt(self,n_trials,obj_func,par_labels,args,lowbnd,uppbnd): # TPE sampler

        
            loss_func = partial(obj_func, model_sim =args[0],X_train = args[1], Y_train = args[2] )  # objective function to be optimized
            
            
            def TPE_space(trials):
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)    #limits the verbosity to warnings
                
                parameters = [trials.suggest_float(par_labels[i],lowbnd[i],uppbnd[i]) for i in range(len(par_labels))] # list of parameters suggested by the sampler
                
                return loss_func(parameters)
            
            sampler = optuna.samplers.TPESampler(seed = 123) # TPE sampler
            
            study  = optuna.create_study(sampler = sampler, direction = 'minimize')
            
            
            
            study.optimize(TPE_space, n_trials = n_trials, show_progress_bar = True, n_jobs= 1, timeout = 3600, catch = (ValueError)) # -1 uses all the CPU CORES
            
            param_list = np.array(list(study.best_trial.params.values())) # list  of parameters found by TPE
            param_dict = study.best_trial.params # dictionary of parameters
            
            return param_dict, param_list
        
        def CMA_opt(self,n_trials,obj_func,par_dict0,par_labels,args,lowbnd,uppbnd): # CMA sampler

        
            loss_func = partial(obj_func, model_sim =args[0],X_train = args[1], Y_train = args[2] )  
            
            
            def CMA_space(trials):
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)    #limits the verbosity to warnings
                
                parameters = [trials.suggest_float(par_labels[i],lowbnd[i],uppbnd[i]) for i in range(len(par_labels))] # list of parameters suggested by the sampler
                
                return loss_func(parameters)
            
            sampler2 = optuna.samplers.CmaEsSampler(x0 = par_dict0, seed = 15, restart_strategy= 'ipop', inc_popsize= 3) # TPE sampler
            
            study2  = optuna.create_study(sampler = sampler2, direction = 'minimize')
            
            
            
            study2.optimize(CMA_space, n_trials = n_trials, show_progress_bar = True, n_jobs= 1, timeout = 3600, catch = (ValueError)) # -1 uses all the CPU CORES
            
            opt_param = np.array(list(study2.best_trial.params.values())) # list  of parameters found by TPE
            # param_dict = study.best_trial.params
            return opt_param 
    
        def fit(self,X_train,Y_train,parameters):
            
            args = (model_sim,X_train,Y_train)
            
            # this function fits the training data 
            self.search_params = self.TPE_opt(100,self.obj_func[0],par_labels,args,lowbnd,uppbnd)
            
            # par_dict0 = self.search_params[0] # dictionaty of parameters for CMA-ES optimizer
            par_list = self.search_params[1] # list of TPE suggested parameters for scipy optimizers
            
            self.least_opt = least_squares(self.obj_func[1], x0 = par_list, method = 'trf',\
                                            bounds = (lowbnd,uppbnd),args = (self.model,X_train,Y_train)) # optimal parameters
            # self.Cma_opt = self.CMA_opt(3000,self.obj_func[0],par_dict0,par_labels,args,lowbnd,uppbnd)
            return self.least_opt.x
       
        def predict(self,X_test,Y_test,fit_param):
    
            return self.model(X_test,Y_test,fit_param)

        
    #%%% Data preparation
    # Split the data in isothermal and adiabatic data
    Y_fold_data = []        # empty list
    X_fold_data = []        # empty list

    isothermal_data = exp_data[exp_data[:,1]== 0]   # extract all isothermal data
    adiabatic_data = exp_data[exp_data[:,1] == 1]    # extract all adiabatic data
    
    for i in Run_isothermal:
        run_data = isothermal_data[isothermal_data[:,0] == i] 
        Y_fold_data.append(run_data)
        X_data = X_input[X_input[:,0] == i] 
        X_fold_data.append(X_data)
        
        
    for j in Run_adiabatic:
        run_data =adiabatic_data[adiabatic_data[:,0] == j] 
        Y_fold_data.append(run_data)
        X_data = X_input[X_input[:,0] == j] 
        X_fold_data.append(X_data)

    # the data is nested in a list that has in each element the experimental data
    
    # convert the data to numpy
    
    Y_kfold_data  = Y_fold_data  # output data as a list indexed by Run number
    
   
    X_kfold_data  = np.concatenate(X_fold_data, axis =0)  # input data
        
    
    Run_kfold  = np.concatenate((Run_isothermal,Run_adiabatic)) # Runs  
    
    

    # create the object to fit and predict data
    model   = kinetic_model(obj_func,model_sim)   
    # fit all the runs for k fold and obtain the parameters
    print("Estimating parameters with all the k-folds data \n")
    fit_fold_par = model.fit(X_kfold_data,np.vstack(Y_kfold_data),par0)
    
    
    #%%% stratified splitter
  
    # mean squared error
    iso_SSR_conc_reg =  []           # SSR of species from isothermal experiments regresion
    iso_SSR_conc_test = []           # SSR of species from isothermal experiments testing
    
    
    ad_SSR_conc_reg =  []           # SSR of species in adiabtic experiments regression
    ad_SSR_conc_test = []           # SSR of species in adiabtic experiments testing
    
    
    SSR_temp_reg =  []              # temperature response from adiabatic experiments in regression
    SSR_temp_test = []              # temperature response from adiabatic experiments in testing
    
 
    training_sets = []             # runs used in each split for training
    testing_sets = []              # runs used in each split for testing 
    
    kfold_par = []                  # parameters estimated during the fold for training in each split
    
    
    
    sfolder = StratifiedKFold(n_splits=splits_no, shuffle= True, random_state = 123) # stratified splitter
    
    print('\n====== Starting stratified '+str(model_label) +'_'+str(splits_no)+' Kfold validation ' + str(round(int(frac_iso*100)/int(frac_ad*100),2))  + ' iso/ad ========\n \n')
    
    
    for train_index, test_index in sfolder.split(Y_kfold_data,Run_labels):
        
        print('\n Training Runs: %s | test Runs: %s' % (np.take(Run_kfold,train_index), np.take(Run_kfold,test_index)))
    
        # train data 
        
        Y_train = np.vstack([Y_kfold_data[i] for i in train_index])   # training experimental data
        X_train = np.take(X_kfold_data,train_index,axis = 0)          # training experimental settings /conditions 
        
        # test data
        Y_test = np.vstack(([Y_kfold_data[i] for i in test_index]))   # validation data
        X_test = np.take(X_kfold_data,test_index,axis = 0)            # validation experimental settings/conditions
    
        #=====================================================
        # save the training and testing runs 
        
        training_sets.append(np.take(Run_kfold,train_index))
        
        
        
        testing_sets.append(np.take(Run_kfold,test_index))
        
        # train the model with training data and fit the parameters
        fit_parameters = model.fit(X_train,Y_train,par0)

        kfold_par.append(fit_parameters) 
        
       
        
        #================================================================================================
        #%% training error, regression stage

        Y_reg     =  model.predict(X_train,Y_train,fit_parameters)
        Y_reg_iso =  Y_reg[0] # isothermal simulation (nx3)
        Y_reg_ad  =  Y_reg[1] # adiabatic simulation (nx4)
        
        
        #extract experimental data for training 
        
        Y_train_iso     = split_data(Y_train,Y_train)[0][:,[4,5,6]]  # isothermal concentrations
        Y_train_ad      = split_data(Y_train,Y_train)[2][:,[4,5,6]]  # adiabatic concentrations
        Y_train_temp    = split_data(Y_train,Y_train)[2][:,7]        # adiabtic temparature
        weight_ad       = split_data(Y_train,Y_train)[2][:,3]        # weight for adiabatic experiments
       

        # isothermal error
        SSR_iso = np.sum((Y_reg_iso-Y_train_iso)**2, axis =0) # isothermal SSR of the concentrations
        iso_SSR_conc_reg.append(SSR_iso)
        
        # adiabtic errors
        
        # concentrations  
        SSR_ad = np.sum(weight_ad[:, np.newaxis]*(Y_train_ad-Y_reg_ad[:,0:3])**2, axis =0)
        ad_SSR_conc_reg.append(SSR_ad)
        
        # temperature errors
        
        SSR_T = np.sum((Y_train_temp-Y_reg_ad[:,3])**2, axis = 0)
        SSR_temp_reg.append(SSR_T)

        #================================================================================================
        #%% prediction error, validation stage
        
        # predict the testing data
       
        Y_pred     = model.predict(X_test,Y_test,fit_parameters)  # simulated data with the X_test
        Y_pred_iso = Y_pred[0]         # isothermal predictions
        Y_pred_ad  = Y_pred[1]         # adiabatic prediction
     
        # extract experimental data for testing  

        Y_test_iso  = split_data(Y_train,Y_test)[1][:,[4,5,6]]  # test isothermal data
        Y_test_temp = split_data(Y_train,Y_test)[3][:,7]        # test adiabtic temparature
        Y_test_ad   = split_data(Y_train,Y_test)[3][:,[4,5,6]]  # test adiabtic temparature
        weight_ad   = split_data(Y_train,Y_test)[3][:,3]        # weight factors for adiabatic concentration 
        
        
        # isothermal errors
        SSR_iso = np.sum((Y_test_iso - Y_pred_iso)**2, axis = 0) # response 1 
        iso_SSR_conc_test.append(SSR_iso)
        
        # temperature and concenctration responses adiabatic experiments
        
        # adiabtic errors
        
        SSR_ad = np.sum(weight_ad[:, np.newaxis]*(Y_test_ad- Y_pred_ad[:,0:3])**2, axis = 0)
        ad_SSR_conc_test.append(SSR_ad)
        
        # temperature error
        SSR_T = np.sum((Y_test_temp- Y_pred_ad[:,3])**2, axis = 0) # temperature
        SSR_temp_test.append(SSR_T)
        
       
        
    CV_k_conc = np.mean(np.vstack((np.vstack(iso_SSR_conc_test),np.vstack(ad_SSR_conc_test))))
    CV_k_temp = np.mean(np.vstack(SSR_temp_test))

        
    iso_scores_reg        = pd.DataFrame(np.vstack(iso_SSR_conc_reg), columns = ['FF','FA','2MF'])
    iso_scores_test       = pd.DataFrame(np.vstack(iso_SSR_conc_test), columns = ['FF','FA','2MF'])
    ad_scores_reg         = pd.DataFrame(np.vstack(ad_SSR_conc_reg), columns = ['FF','FA','2MF'])
    ad_scores_test        = pd.DataFrame(np.vstack(ad_SSR_conc_test), columns = ['FF','FA','2MF'])
    temp_scores_reg       = pd.DataFrame(np.vstack(SSR_temp_reg), columns = ['Temp K'])
    temp_scores_test      = pd.DataFrame(np.vstack(SSR_temp_test), columns = ['Temp K'])
    kfold_par             = pd.DataFrame(np.transpose(np.vstack(kfold_par)), index= par_labels) # parameters
    all_fold_par          = pd.DataFrame(np.transpose(fit_fold_par), index = par_labels)
    training_sets         = pd.DataFrame(np.vstack(training_sets))
    testing_sets          = pd.DataFrame(np.vstack(testing_sets))
   
    #%% save data in excel files
    with pd.ExcelWriter('CV_'+str(splits_no) + str(model_label)+'_ratio iso_ad_'+ \
                        str(round(int(frac_iso*100)/int(frac_ad*100),1)) +'.xlsx' ) as writer: # write data in a excel file
        
        training_sets.to_excel(writer, sheet_name= "training runs")
        testing_sets.to_excel(writer,sheet_name = "testing runs")
        iso_scores_reg.to_excel(writer,sheet_name="SSR_iso_train")
        iso_scores_test.to_excel(writer,sheet_name="SSR_iso_test")
        
        
        ad_scores_reg.to_excel(writer,sheet_name='SSR_ad_train')
        ad_scores_test.to_excel(writer,sheet_name='SSR_ad_test')
        
        temp_scores_reg.to_excel(writer,sheet_name = "SSR_temp_train")
        temp_scores_test.to_excel(writer,sheet_name = "SSR_temp_test")
        
        kfold_par.to_excel(writer,sheet_name = 'kfold_parameters')
        all_fold_par.to_excel(writer, sheet_name= "parameters with all k-fold data")
    
    CV_score = np.mean((CV_k_conc,CV_k_temp))
    
    print("StratifiedKFold done")
    
    return   CV_k_conc,CV_k_temp,CV_score