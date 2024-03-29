import numpy as np
import Models
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm

def obj_modelo_analitico_P (params, data, t):
    
    sim = Models.modelo_analitico(t, params, Rsquare=False, data=None)
    res = data - sim[3]
    
    return res

def obj_modelo_analitico_SP (params, data, t):
    
    sim = Models.modelo_analitico(t, params, Rsquare=False, data=None)
    
    res_sa = data[0]['concentração'] - sim[0]
    res_sd = data[1]['concentração'] - sim[3]
    
    res_total = np.sqrt(np.square(res_sa)) + np.sqrt(np.square(res_sd))
    
    return res_total

def obj_modelo_analitico_SPI (params, data, t):
    
    sim = Models.modelo_analitico(t, params, Rsquare=False, data=None)
       
    res_sa = data[0]['concentração'] - sim[0]
    res_sc = data[1]['concentração'] - sim[2]
    res_sd = data[2]['concentração'] - sim[3]
       
    res_total = np.sqrt(np.square(res_sa)) + np.sqrt(np.square(res_sc)) + np.sqrt(np.square(res_sd))
    
    return res_total
    

def obj_wrapper_model_2 (params, data, t, x, t_compute):
   
    sim = Models.wrapper_model_2(t, x, params, t_compute, Rsquare=False, data=None)
    res = data - sim[4]
   
    
    return res
 
def obj_wrapper_model_2_SP (params, data, t, x, t_compute):
    
    sim = Models.wrapper_model_2(t, x, params, t_compute, Rsquare=False, data=None)

    res_SA = data[0]['concentração'] - sim[6]
    res_SD = data[1]['concentração'] - sim[4]
    
    res_total = np.sqrt(np.square(res_SA)) + np.sqrt(np.square(res_SD))
    
    return res_total

def obj_wrapper_model_2_SPI (params, data, t, x, t_compute):
    
    sim = Models.wrapper_model_2(t, x, params, t_compute, Rsquare=False, data=None)

    res_SA = data[0]['concentração'] - sim[6]
    res_SC = data[1]['concentração'] - sim[3]
    res_SD = data[2]['concentração'] - sim[4]

    
    res_total = np.sqrt(np.square(res_SA)) + np.sqrt(np.square(res_SD)) + np.sqrt(np.square(res_SC))
    
    return res_total

def obj_model_monod_P (params, data, t, x, t_compute):
    
    init = [x[0], params['X0'].value, x[2]]
    print(init)
    sim = Models.model_monod_sr(t, init, params, t_compute, Rsquare=False, data=None)
    
    res_P = data[1]['concentração'] - sim[2]
    
    return res_P


def obj_model_monod_S (params, data, t, x, t_compute):
    
    
    sim = Models.model_monod_sr(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]
    
    return res_S


def obj_model_monod_SP (params, data, t, x, t_compute):
    
    sim = Models.model_monod_sr(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]   
    res_P = data[1]['concentração'] - sim[2]
    
    res_total = np.sqrt(np.square(res_S)) + np.sqrt(np.square(res_P))
    
    return res_total

def obj_P_model_monod_sr_x (params, data, t, x, t_compute):
    
    sim = Models.model_monod_sr_x(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_P = data[1]['concentração'] - sim[2]
    
    return res_P

def obj_S_model_monod_sr_x (params, data, t, x, t_compute):
    
    sim = Models.model_monod_sr_x(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]
    
    return res_S

def obj_SP_model_monod_sr_x (params, data, t, x, t_compute):
    
    sim = Models.model_monod_sr_x(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]
    res_P = data[1]['concentração'] - sim[2]
    
    res_total = np.sqrt(np.square(res_S)) + np.sqrt(np.square(res_P))
    
    return res_total


def obj_P_monod_hyd (params, data, t, x, t_compute):
    
    sim = Models.monod_hyd(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_P = data[1]['concentração'] - sim[4]
    
    return res_P


def obj_SP_monod_hyd (params, data, t, x, t_compute):
    
    sim = Models.monod_hyd(t, x, params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]
    res_P = data[1]['concentração'] - sim[4]
    
    res_total = np.sqrt(np.square(res_S)) + np.sqrt(np.square(res_P))
    
    return res_total


def obj_monod_hyd_alt (params, data, t, x, t_compute):
    
    print(params.valuesdict())
    sAr_0 = x[0]
    sAl_0 = x[1]
    sB_0  = x[2]
    sC_0  = x[3]
    P_0   = x[5]
    sim = Models.monod_hyd_alt(t, [sAr_0, sAl_0, sB_0, sC_0, params['X_0'].value, P_0], params, t_compute, Rsquare=False, data=None)
    
    res_S = data[0]['concentração'] - sim[0]
    res_I = data[1]['concentração'] - sim[4]
    res_P = data[2]['concentração'] - sim[6]
    
    res_total = np.sqrt(np.square(res_S)) + np.sqrt(np.square(res_I)) + np.sqrt(np.square(res_P)) 
    
    return res_total