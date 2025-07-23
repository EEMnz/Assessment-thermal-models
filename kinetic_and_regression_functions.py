# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:31:38 2024

@author: ERNY Encarnacion Munoz
"""

# Reaction mechanism
# FF + H2 ==>  FA   R1
# FA+ H2   ==> 2-MF  R2 


#import packages
import numpy as np
from scipy.integrate import solve_ivp
import numdifftools as nd
# from numba import jit



# thermodynamic properties
# enthalpies
H1= -4.184*4*1e3  #J/mol https://doi.org/10.1021/jp306596d
H2= -68.84*1e3    #J/mol https://doi.org/10.1021/acs.energyfuels.0c01598

#water CP coefficients ref: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=2&Type=JANAFL&Plot=on#JANAFL

#valid from 298-500 K

MFF = 96.0846             #g/mol Furfural
MW  = 18.02               #g/mol water
MFA = 98.1                #g/mol furfuryl alcohol
M2MF = 82.1               #g/mol methyl furan

# water properties
rho_H20 = 1000 # kg/m3
                      # Cp https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Type=JANAFL&Plot=on#JANAFL
A = -203.6060
B = 1523.290
C = -3196.413
D = 2474.455
E = 3.855326

# solubility parameters
kla    = 1.88E-02       # absortion constant 
He_ref = 7.80*1e-6      # henry constant of H2  in water at 298.15K,  mol.m-3/pa  ref doi: Compilation of Henry’s law constants (version 5.0.0) 
dH_sol_R = 530          # https://henrys-law.org/henry/casrn/1333-74-0 at 298.15 K
Tref = 120 + 273.15     # K reference temp for rate constants
Tref_He = 25 + 294.15   # K reference temp for henry


#%% material balances
# @jit(nopython = True)
def model_LH1(t,c,arg,par_1):
    
    # Langmuir Hinshelwood with hydrogen dissociation (LH1) 
    
    # states variables concentration in mol/m3
    FF   = c[0] # furfural
    FA   = c[1] # furfuryl alcohol
    MF   = c[2] # 2 methyl furan
    W    = c[3] # water
    H2   = c[4] # hydrogen
    Temp = c[5] # temperature in K
    
    #reaction volume and mass
    Vrx   = arg[3]
    mrx   = arg[4]
    # arguments
    alpha = arg[0]           # thermal_mode # 0 for isothermal, 1 for adiabatic
    wcat  = arg[1]*1e-3/Vrx  # Catalyst loading in kg/m3
    p_H2  = arg[2]*1e5       # hydrogen pressure in pa    
   
    T     = Temp/1000        # Temperature used for Cp Correlation

    # parameters to estimate
    ln_k01 = par_1[0]
    ln_k02 = par_1[1]
    Ea_RT1 = par_1[2]
    Ea_RT2 = par_1[3]
    KFF  = par_1[4]*1e-3 #m3/mol
    KFA  = par_1[5]*1e-3 #m3/mol
    K2MF = par_1[6]*1e-3 #m3/mol
    KH   = par_1[7]*1e-3 #m3/mol
    KW   = par_1[8]*1e-3 #m3/mol

    # Henry constants 
    He_T = He_ref*np.exp((dH_sol_R)*(1/Temp - 1/Tref_He))    # van't Hoff equation for Henrys depence with temperature
    CH_eq = He_T*p_H2                                        # concentration of hydrogen at equilibrium
    Cp = (A + B*T + C*T**2 + D*T**3 + E/T**2)*1e3/MW         # J/kg*K
    
    # rate constants
    k1 = np.exp(ln_k01 + Ea_RT1*(1 -(Tref/Temp)))
    k2 = np.exp(ln_k02 + Ea_RT2*(1 -(Tref/Temp)))


    #free sites expression
    phi_1 = ((1+KFF*FF+KFA*FA+K2MF*MF+W*KW)+(KH*H2)**0.5)**(-1)
    
    #rate expresions
    
    R1=k1*H2*KH*FF*KFF*wcat*(phi_1**3)
    R2=k2*H2*KH*FA*KFA*wcat*(phi_1**3)


    # ODE system to integrate
    dFFdt   = -R1
    dFAdt   = R1-R2
    dMFdt   = R2
    dWdt    = R2
    dH2dt   = kla*(CH_eq-H2)-R1-R2
    dTempdt = alpha*(-R1*H1-R2*H2)*Vrx/(mrx*Cp)

    
    return np.array([dFFdt, dFAdt, dMFdt, dWdt, dH2dt, dTempdt])


 
# @jit(nopython = True)
def model_NCLH2(t,c,arg,par_2):

    # non-competitive Langmuir Hinshelwood with hydrogen dissociation (NCLH2) 
    
    # states variables concentration in mol/m3
    FF   = c[0] # furfural
    FA   = c[1] # furfuryl alcohol
    MF   = c[2] # 2 methyl furan
    W    = c[3] # water
    H2   = c[4] # hydrogen
    Temp = c[5] # temperature in K
    
    #reaction volume and mass
    Vrx   = arg[3]
    mrx   = arg[4]
    # arguments
    alpha = arg[0]           # thermal_mode # 0 for isothermal, 1 for adiabatic
    wcat  = arg[1]*1e-3/Vrx  # Catalyst loading in kg/m3
    p_H2  = arg[2]*1e5       # hydrogen pressure in pa    
   
    T     = Temp/1000        # Temperature used for Cp Correlation

    # parameters to estimate
    ln_k01 = par_2[0]
    ln_k02 = par_2[1]
    Ea_RT1 = par_2[2]
    Ea_RT2 = par_2[3]
    KFF  = par_2[4]*1e-3  #m3/mol
    KFA  = par_2[5]*1e-3
    K2MF = par_2[6]*1e-3
    KH   = par_2[7]*1e-3
    KW   = par_2[8]*1e-3

    
    # Henry constants 
    He_T = He_ref*np.exp((dH_sol_R)*(1/Temp - 1/Tref_He))    # van't Hoff equation for Henrys depence with temperature
    CH_eq = He_T*p_H2                                        # concentration of hydrogen at equilibrium
    Cp = (A + B*T + C*T**2 + D*T**3 + E/T**2)*1e3/MW         # J/kg*K
    
    # rate constants
    k1 = np.exp(ln_k01 + Ea_RT1*(1 -(Tref/Temp)))  
    k2 = np.exp(ln_k02 + Ea_RT2*(1 -(Tref/Temp)))  

    #free sites expression
    phi_1 = (1+KFF*FF+KFA*FA+K2MF*MF+W*KW)**(-1) # free sites for FF componets
    phi_2 = (1+(KH*H2)**0.5)**(-1) # free sites for hydrogen
    
    #rate expresions
    
    R1=k1*KH*H2*KFF*FF*wcat*phi_1*(phi_2**2)
    R2=k2*KH*H2*KFA*FA*wcat*phi_1*(phi_2**2)
    

    # ODE system to integrate
    dFFdt   = -R1
    dFAdt   = R1-R2
    dMFdt   = R2
    dWdt    = R2
    dH2dt   = kla*(CH_eq-H2)-R1-R2
    dTempdt = alpha*(-R1*H1-R2*H2)*Vrx/(mrx*Cp)

    
    return np.array([dFFdt, dFAdt, dMFdt, dWdt, dH2dt, dTempdt])

# @jit(nopython = True)
def model_ELR3(t,c,arg,par_3):

    # Eley Rideal mechanisms 
    
    # states variables concentration in mol/m3
    FF   = c[0] # furfural
    FA   = c[1] # furfuryl alcohol
    MF   = c[2] # 2 methyl furan
    W    = c[3] # water
    H2   = c[4] # hydrogen
    Temp = c[5] # temperature in K
    
   #reaction volume and mass
    #reaction volume and mass
    Vrx   = arg[3]
    mrx   = arg[4]
    # arguments
    alpha = arg[0]           # thermal_mode # 0 for isothermal, 1 for adiabatic
    wcat  = arg[1]*1e-3/Vrx  # Catalyst loading in kg/m3
    p_H2  = arg[2]*1e5       # hydrogen pressure in pa    
   
    T     = Temp/1000        # Temperature used for Cp Correlation
    

    # parameters to estimate
    ln_k01 = par_3[0]
    ln_k02 = par_3[1]
    Ea_RT1 = par_3[2]
    Ea_RT2 = par_3[3]
    KH   = par_3[4]*1e-3 #m3/mol

    # Henry constants 
    He_T = He_ref*np.exp((dH_sol_R)*(1/Temp - 1/Tref_He))    # van't Hoff equation for Henrys depence with temperature
    CH_eq = He_T*p_H2                                        # concentration of hydrogen at equilibrium
    Cp = (A + B*T + C*T**2 + D*T**3 + E/T**2)*1e3/MW         # J/kg*K
    
    # rate constants
    k1 = np.exp(ln_k01 + Ea_RT1*(1 -(Tref/Temp)))
    k2 = np.exp(ln_k02 + Ea_RT2*(1 -(Tref/Temp)))
   


    #free sites expression # only adsorption of hydrogen
    phi_1 = (1+(KH*H2)**0.5)**(-1)                           # free sites 
 
    #rate expresions   
    R1=k1*KH*H2*FF*wcat*(phi_1**2)
    R2=k2*KH*H2*FA*wcat*(phi_1**2)


    # ODE system to integrate
    dFFdt   = -R1
    dFAdt   = R1 - R2
    dMFdt   = R2
    dWdt    = R2
    dH2dt   = kla*(CH_eq-H2)-R1-R2
    dTempdt = alpha*(-R1*H1-R2*H2)*Vrx/(mrx*Cp)

    
    return np.array([dFFdt, dFAdt, dMFdt, dWdt, dH2dt, dTempdt])


#%% model ODE integration
class Model_simulation:
    
    """ class with function to simulate the kinetic model
    rx_model (func) : material balance function
    
    """
    
    def __init__(self, rx_model):
        
        self.model = rx_model # kinetic model material balance
        pass
    
    # models integration over initial experimental conditions.
    def simulate(self,X_data,Y_data,parameters):
    
        """ 
        the data has to be in numpy format and ordered
        
        # initial conditions (1D or 2D np.array) :  experimantal settings to simulate
        # par (1D array) : model's parameter
        # Y_data = experimental data

        """ 
        
        init_cond =  X_data # initial conditions
        
        t_eval = Y_data[:,0:3] # extract the colums that have the run,thermal mode,time
        
        t_sp = [] # empty list
        runs_label = init_cond[:,0] # extract the runs number from the initial conditions matrix
                                    
        for i in runs_label :
            t_sp.append(t_eval[t_eval[:,0] ==i]) # evaluation time in a nested list index by the number of runs
        
        
        
        par = *parameters, # unpack the parameters
        
        isothermal_out = [] # empty list to save the concentrations in isothermal mode
        adiabatic_out  = [] # empty list to save concentration and temperature responses
    
        

        if init_cond.ndim == 2 :            # for a matrix of experimantal conditions
            arg_data = init_cond[:,[1,6,8]] # thermal mode, cat, press  
            n_runs = len(init_cond)
            
            for i in range(n_runs):         # this loop execute the models for each Run i
                
                par = par #  parameters values for model 1 LH1
                
                # calculation of reaction conditions
                Vrx   = (np.sum(init_cond[i,2:6])/1000)/(0.77*rho_H20 +0.22*1160)         # reaction volume in m3
                mrx   = np.sum(init_cond[i,2:6])/1000                                     # reaction total mass kg
                c0    = (init_cond[i,2:6]/[MFF,MFA,M2MF,MW])/Vrx                          # initial concentrations of species mol/m3
                T_0   = [init_cond[i,7] + 273.15]                                          # initial reaction temperature K 
                
                # add initial hydrogen conc, set to 0 because at that time stirring has not started
                cH20 = [0]
                c0   = np.hstack((c0,cH20))
                c0   = np.concatenate((c0,T_0),axis=0)                 # initial concentrations of species mol/m3 and initial temperature in K
                arg  = np.append(arg_data[i,:],[Vrx,mrx])              # arguments of the function: cat, temp, press, volume, mass rx
        
                
                if (init_cond[i,1] == 0 and np.unique(t_sp[i][:,1])==0) : # solver for isothermal experiments
                    
                     # reaction time
                    t_iso = t_sp[i][:,2]  # evaluation times in sec
                    #ODE integrator
                    sol_EDO = solve_ivp(self.model,t_span = [0,36500], y0 = c0,method = 'BDF',t_eval = t_iso,rtol = 1e-6, atol= 1e-8, args = (arg,par))
                    isothermal_out.append(sol_EDO.y.T[:,[0,1,2]]) # concentration responses (mol/m3)
        
                
                elif (init_cond[i,1] == 1 and np.unique(t_sp[i][:,1])==1): # solver for adiabatic experiments 
                    
                    # reaction time
                    t_ad = t_sp[i][:,2] # evaluation times in sec
                
                    #ODE integrator
                    sol_EDO = solve_ivp(self.model,t_span = [0,36500], y0 = c0,method = 'BDF',t_eval = t_ad ,rtol = 1e-6, atol= 1e-8, args = (arg,par))
                    adiabatic_out.append(sol_EDO.y.T[:,[0,1,2,5]]) # temperature and concenctration responses (mol/m3, K)
                    
            if(init_cond[:,1] == 1).any() == True and (init_cond[:,1] == 0).any() == True : # mix adiabatic, isothermal experiments
            
                isothermal_out = np.concatenate(isothermal_out, axis = 0)
                adiabatic_out  = np.concatenate(adiabatic_out, axis = 0)
                
            elif (init_cond[:,1] == 1).all() == False: # no adiabatic experiments
                isothermal_out = np.concatenate(isothermal_out, axis = 0)
                adiabatic_out  = [0]
                
            elif (init_cond[:,1] == 0).all() == False: # no isothermal experiments
                isothermal_out = [0]
                adiabatic_out  = np.concatenate(adiabatic_out, axis = 0)
                
            return isothermal_out,adiabatic_out # arrays with the responses
       

#%% data splitter

def split_data(Y_train,Y_test): # data input has to be in numpy

    """ split the experimental data for training and testing, to use only for training  make Y_train = Ytest"""
    # isothermal
    
    train_index_iso = np.argwhere(np.vstack(Y_train)[:,1]== 0) # extract index of isothermal data
    Y_train_iso = np.take(np.vstack(Y_train),train_index_iso.T[0],axis =0) # isothermal data for training
    
    # adiabatic data
    train_index_ad = np.argwhere(np.vstack(Y_train)[:,1]== 1) # extract index of adiabatic data
    Y_train_ad = np.take(np.vstack(Y_train),train_index_ad.T[0],axis =0) # isothermal data for training
    
    # testing data
    #isothermal
    test_index_iso = np.argwhere(np.vstack(Y_test)[:,1]== 0) # extract index of isothermal data
    Y_test_iso = np.take(np.vstack(Y_test),test_index_iso.T[0],axis =0) # isothermal data for testing
    
    # adiabatic data
    test_index_ad = np.argwhere(np.vstack(Y_test)[:,1]== 1) # extract index of adiabatic data
    Y_test_ad = np.take(np.vstack(Y_test),test_index_ad.T[0],axis =0) # isothermal data for testing
    
    return Y_train_iso,Y_test_iso,Y_train_ad,Y_test_ad   


#%% Regression functions

def SSE_func(parameters,model_sim,X_train,Y_train): # input data has to be in numpy
        
    """ SSE function with normalized concentrations from 0 to 1 
    
        parameters : list of parameters
        model_sim: function to simulate the kinetic model
        X_train : numpy array with the experimental conditions to be simulated
        Y_train : numpy array with the experimental data
    
    """
    
    from sklearn.preprocessing import minmax_scale       
    
    
    Y_test = Y_train
      
    sim_isothermal, sim_adiabatic = model_sim(X_train,Y_train,parameters)
    
    
    # calculate the residuals
    if (Y_train[:,1] == 0).any() == True and (Y_train[:,1]==1).any() == True : # for both isothermal and adiabatic experiment
    
        
        weight_ad = split_data(Y_train,Y_test)[2][:,3] # weight for adiabatic experiments
        index_ad   = np.argwhere(weight_ad>0)
    
        # scale the data between zero and one
        
        sim_isothermal =  minmax_scale(sim_isothermal, axis = 0)      
        sim_adiabatic  =  minmax_scale(sim_adiabatic,axis=0)
    
        # extract training experimental data and scale between 0 and 1
    
        exp_isothermal = minmax_scale(split_data(Y_train,Y_test)[0][:,[4,5,6]], axis = 0)  # isothermal concentrations mol/m3
        exp_adiabatic  = minmax_scale(split_data(Y_train,Y_test)[2][:,[4,5,6]] , axis = 0) # adiabatic concentrations mol/m3
        exp_temp_ad    = minmax_scale(split_data(Y_train,Y_test)[2][:,7], axis = 0)        # adiabtic temparature
        
        
        res_iso = (exp_isothermal - sim_isothermal).ravel(order = 'F') # isothermal residuals
        
        res_ad  = (np.take(exp_adiabatic,index_ad.T[0],axis = 0) - np.take(sim_adiabatic[:,0:3],index_ad.T[0],axis =0)).ravel(order = 'F')  # adiabatic residuals concentration
        
        res_temp = (exp_temp_ad-sim_adiabatic[:,3]).ravel(order = 'F')  # temperature residuals
        
        residuals = np.concatenate((res_iso,res_ad,res_temp), axis = 0)
    
    # for isothermal  experiments only
    elif (Y_train[:,1] == 1).all() == False:
        
        sim_isothermal =  minmax_scale(sim_isothermal, axis = 0)    
        
        exp_isothermal = minmax_scale(split_data(Y_train,Y_test)[0][:,[4,5,6]], axis = 0)  # isothermal concentrations mol/m3
        res_iso = (exp_isothermal - sim_isothermal).ravel(order = 'F') # isothermal residuals
        residuals = res_iso
    
    return residuals
  
    
def Det_criterion(parameters,model_sim,X_train,Y_train): # input data has to be in numpy

    """ computes the determinant 
    
        parameters : list of parameters
        model_sim: function to simulate the kinetic model
        X_train : numpy array with the experimental conditions to be simulated
        Y_train : numpy array with the experimental data
        
    """

    Y_test = Y_train
     
    sim_isothermal, sim_adiabatic = model_sim(X_train,Y_train,parameters)
    
    if (Y_train[:,1] == 0).any() == True and (Y_train[:,1]==1).any() == True : # for both isothermal and adiabatic experiment
    
    
        weight_ad = split_data(Y_train,Y_test)[2][:,3] # weight for adiabatic experiments
        
        # index_ad   = np.argwhere(weight_ad>0)
        
        # extract training experimental data
    
        exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations  mol/m3
        exp_adiabatic  = split_data(Y_train,Y_test)[2][:,[4,5,6]]  # adiabatic concentrations  mol/m3
        exp_temp_ad    = split_data(Y_train,Y_test)[2][:,7]        # adiabtic temparature
    
        # compute the residuals
        
        res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
        res_conc_ad = weight_ad[:, np.newaxis]*(exp_adiabatic-sim_adiabatic[:,0:3]) # adiabatic residuals residuals FF, 2MF, MTHF 
        res_temp = (exp_temp_ad-sim_adiabatic[:,3])  # temperature residuals from adiabatic experiments 
        res_ad = np.c_[res_conc_ad,res_temp] # (nx4) adiabatic responses
        
        # compute the residual matrices
        
        v_theta_iso = res_iso.T@res_iso # 3x3 
        v_theta_ad  = res_ad.T@res_ad # 4x4
      
        det_iso = np.linalg.slogdet(v_theta_iso)
        det_ad = np.linalg.slogdet(v_theta_ad)
        det_crit = det_iso[1] + det_ad[1]
        
        return det_crit
        
      
    elif (Y_train[:,1] == 1).all() == False: # for isothermal experiments only
    
        exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations mol/m3
        res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
        
        v_theta = res_iso.T@res_iso
    
        det = np.linalg.slogdet(v_theta)
        
        det_crit = det[1]
       
        
        return  det_crit

        
#%% statistical parameters            

           
class Statistical_inferece :
    
    """ Class with methods to compute the determinant and residuals vector"""
    def __init__(self):
        
        pass
                   
    def res_vector(self, parameters,model_sim,X_train,Y_train): # input data has to be in numpy

    
        """ this function returns the residuals
        
        parameters : list of parameters
        model_sim: function to simulate the kinetic model
        X_train : numpy array with the experimental conditions to be simulated
        Y_train : numpy array with the experimental data
        
        """
        Y_test = Y_train
      
        sim_isothermal, sim_adiabatic = model_sim(X_train,Y_train,parameters)
    
    
        # calculate the residuals
        if (Y_train[:,1] == 0).any() == True and (Y_train[:,1]==1).any() == True : # for both isothermal and adiabatic experiment
        
            
            weight_ad = split_data(Y_train,Y_test)[2][:,3] # weight for adiabatic experiments
            index_ad   = np.argwhere(weight_ad>0)
        
            # extract training experimental data and scale between 0 and 1
        
            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations  mol/m3
            exp_adiabatic  = split_data(Y_train,Y_test)[2][:,[4,5,6]]  # adiabatic concentrations mol/m3
            exp_temp_ad    = split_data(Y_train,Y_test)[2][:,7]             # adiabtic temparature
            
            res_iso = (exp_isothermal - sim_isothermal).ravel(order = 'F') # isothermal residuals
            
            res_ad  = (np.take(exp_adiabatic,index_ad.T[0],axis = 0)-np.take(sim_adiabatic[:,0:3],index_ad.T[0],axis =0)).ravel(order = 'F')  # adiabatic residuals concentration
            
            res_temp = (exp_temp_ad-sim_adiabatic[:,3]).ravel(order = 'F') 
            
            residuals = np.concatenate((res_iso,res_ad,res_temp), axis = 0)
          
        
        # for isothermal  experiments only
        elif (Y_train[:,1] == 1).all() == False:

            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations  mol/m3
            res_iso = (exp_isothermal - sim_isothermal).ravel(order = 'F') # isothermal residuals
            residuals = res_iso
             
        
        return residuals

         
    def MVLSQ(self, parameters,model_sim,X_train,Y_train): # input data has to be in numpy

        """ computes multivariate least squares

            parameters : list of parameters
            model_sim: function to simulate the kinetic model
            X_train : numpy array with the experimental conditions to be simulated
            Y_train : numpy array with the experimental data
            
        """     

        Y_test = Y_train
         
        sim_isothermal, sim_adiabatic = model_sim(X_train,Y_train,parameters)
        
        if (Y_train[:,1] == 0).any() == True and (Y_train[:,1]==1).any() == True : # for both isothermal and adiabatic experiment
        
        
            weight_ad = split_data(Y_train,Y_test)[2][:,3] # weight for adiabatic experiments
            
            # index_ad   = np.argwhere(weight_ad>0)
            
            sim_isothermal =  sim_isothermal    
            sim_adiabatic  =  sim_adiabatic 
        
            # extract training experimental data and scale between 0 and 1
        
            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations 
            exp_adiabatic  = split_data(Y_train,Y_test)[2][:,[4,5,6]]  # adiabatic concentrations
            exp_temp_ad    = split_data(Y_train,Y_test)[2][:,7]        # adiabtic temparature
         
            
         # compute the residuals
            
            res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
            res_conc_ad = weight_ad[:, np.newaxis]*(exp_adiabatic-sim_adiabatic[:,0:3]) # adiabatic residuals residuals FF, FA, 2MF 
            res_temp = (exp_temp_ad-sim_adiabatic[:,3])  # temperature residuals from adiabatic experiments 
            res_ad = np.c_[res_conc_ad,res_temp] # (nx4) adiabatic responses
            
            # compute the error covariance matrix
            cov_iso =   np.linalg.pinv(res_iso.T@res_iso/(len(Y_train)-len(parameters))) #3x3
            # cov_iso = np.diag(np.diag(cov_iso))
            
            cov_ad  =   np.linalg.pinv(res_ad.T@res_ad /(len(Y_train)-len(parameters))) # 4x4
            # cov_ad = np.diag(np.diag(cov_iso))
            
            z_iso = [] # empty list
            z_ad = [] #  empty list
            
            for u in range(len(res_iso)): # assuminng constants covariance matrix "homecedastic"
               
               z_iso_u  = res_iso[u,:][np.newaxis,:]@cov_iso@res_iso[u,:][np.newaxis,:].T # 1xm @mxm @mx1 = 1D
              
               z_iso.append(z_iso_u) 
            
            for u in range(len(res_ad)):
                z_ad_u   =   res_ad[u,:][np.newaxis,:]@cov_ad@res_ad[u,:][np.newaxis,:].T # 1xm @mxm @mx1 = 1D
                z_ad.append(z_ad_u)
            
            z_iso = np.vstack(np.asarray(z_iso))
            z_ad  =np.vstack(np.asarray(z_ad))
            z_t = np.vstack((z_iso,z_ad))  # total residuals vector 
            
            return np.sum(z_t.flatten(order = 'F'), axis =0)
            
          
        elif (Y_train[:,1] == 1).all() == False: # for isothermal experiments only
            
            sim_isothermal = sim_isothermal     
        
            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations 
            res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
            
            # compute the residual matrices
            cov_iso =   np.linalg.pinv(res_iso.T@res_iso/(len(Y_train)-len(parameters))) #3x3
            # cov_iso = np.diag(np.diag(cov_iso))
            
            z_iso = [] # empty list
           
            for u in range(len(res_iso)):
               
               z_iso_u  =   res_iso[u,:][np.newaxis,:]@cov_iso@res_iso[u,:][np.newaxis,:].T # 1xm @mxm @mx1 = 1D
               z_iso.append(z_iso_u) 
            
            z_t = np.vstack(np.asarray(z_iso))  # residuals vector 
           
            return   np.sum(z_t.flatten(order = 'F'), axis = 0)
            



    def det_calculation(self, parameters,model_sim,X_train,Y_train): # calculate the determinant
        
        """  HPD region by using a multivariate t dist in |v(theta)|

        parameters : list of parameters
        model_sim: function to simulate the kinetic model
        X_train : numpy array with the experimental conditions to be simulated
        Y_train : numpy array with the experimental data
        
        """   
    
        Y_test = Y_train
        sim_isothermal, sim_adiabatic = model_sim(X_train,Y_train,parameters)
        
        # for isothermal and adiabatic experiments
        if (Y_train[:,1] == 0).any() == True and (Y_train[:,1]==1).any() == True : # for both isothermal and adiabatic experiment
        
        
            weight_ad = split_data(Y_train,Y_test)[2][:,3] # weight for adiabatic experiments
           
            
            # extract training experimental data
    
            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]] # isothermal concentrations mol/m3
            exp_adiabatic  = split_data(Y_train,Y_test)[2][:,[4,5,6]] # adiabatic concentrations  mol/m3
            exp_temp_ad    = split_data(Y_train,Y_test)[2][:,7]       # adiabtic temparature
        
            # compute the residuals
            
            res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
            res_conc_ad = weight_ad[:, np.newaxis]*(exp_adiabatic-sim_adiabatic[:,0:3]) # adiabatic residuals residuals FF, 2MF, MTHF 
            res_temp = (exp_temp_ad-sim_adiabatic[:,3])  # temperature residuals from adiabatic experiments 
            res_ad = np.c_[res_conc_ad,res_temp] # (nx4) adiabatic responses
            
            # compute the residual matrices
            
            v_theta_iso = res_iso.T@res_iso # 3x3 
            v_theta_ad  = res_ad.T@res_ad # 4x4
           
            det_iso = np.linalg.det(v_theta_iso)
            det_ad = np.linalg.det(v_theta_ad)
            det_crit = det_iso + det_ad
            
            return det_crit
            
        elif (Y_train[:,1] == 1).all() == False: # for isothermal experiments only
        
            exp_isothermal = split_data(Y_train,Y_test)[0][:,[4,5,6]]  # isothermal concentrations mol/m3
            res_iso = (exp_isothermal - sim_isothermal)  # isothermal residuals FF, 2MF, MTHF
            
            v_theta = res_iso.T@res_iso
            det_iso = np.linalg.det(v_theta)
            
            det_crit = det_iso
           
            
            return  det_crit

        
    



