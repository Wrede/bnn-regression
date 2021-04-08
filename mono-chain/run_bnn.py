#!/usr/bin/env python
# coding: utf-8

from LV_model import uniform_prior, uniform_prior_dens, simulator, lotka_volterra, newData
import logging, dask, scipy, pickle, absl, sklearn
from bcnn_model import Regressor
from sklearn.model_selection import train_test_split 
from gaussian import Gaussian
import numpy as np
import pandas as pd
import time


def minmax_normalize(data, min_=None, max_=None, axis=0):
    """
        min max normalize (0 - ~1) of scale given min_ and max_
        data = data to be scales
        min_ = lower scale
        max_ = upper scale
        axis = which numpy axis to adress the scaling.
    """
    data_ = np.copy(data)
    if min_ is None:        
        min_ = np.min(data_,axis=axis)
        min_ = np.expand_dims(min_,axis=0)
    
    if max_ is None:
        max_ = np.max(data_,axis=axis)
        max_ = np.expand_dims(max_,axis=0)
    
    min_[np.where(min_ == max_)] = 0

    return (data_ - min_) / (max_ - min_)

def checkprior(theta):
    """
       If sample in theta is outside of prior, then remove it from the sample.
       theta = random samples, shape:(N,3)
    """
    flag = uniform_prior_dens(theta) > 0
    print(f'len(flag): {sum(flag)}')
    print(f'total: {theta.shape}')

    return theta[flag,:]

def run_bnn(total_runs=10, num_rounds=6, num_data_per_round = 1000, num_monte_carlo=500, 
                                                                    batch_size=128, 
                                                                    ID='test', 
                                                                    seed=None,
                                                                    epochs=100,
                                                                    correction=True,
                                                                    agg_data=True,
                                                                    infer_pdf=True):
    np.random.seed(seed)
    save_folder = ID
    Ndata = num_data_per_round # number of model samples
    initial_prior = uniform_prior 
    #initial_prior = triangle_prior
    result_posterior = [] #store posterior samples per round
    result_proposal_prior = [] #store m, S of gaussian proposal
    store_time = [] # time per round without simulation

    target_theta = np.log([[1.0,0.1, 0.05]])
    target_ts = np.load('target_ts.npy') #Load the observation
    target_ts = np.transpose(target_ts, (0,2,1))
    notscaled = False
    print("target shape:", target_ts.shape)

    try:
    
        for run in range(total_runs):
            print(f'starting run {run}')
            theta = []
            theta_tot = []
            proposals = []
            data_tot = []
            theta_tot = []
            time_ticks = []
            theta.append(initial_prior(Ndata))
            retrain = False
            min_ = None
            max_ = None

            for i in range(num_rounds):
                
                print(f'starting round {i}')

                # generate new data
                if len(theta[i] > 0):
                    data_ts, data_thetas = newData(theta[i], toexp=True)
                    data_ts = np.transpose(data_ts, (0,2,1))
                    #if min_ is None:
                    #    min_ = np.min(data_ts,axis=0)
                    #    min_ = np.expand_dims(min_,axis=0)
                    #if max_ is None:
                    #    max_ = np.max(data_ts,axis=0)
                    #    max_ = np.expand_dims(max_,axis=0)
                    #if not notscaled:
                    #    target_ts = minmax_normalize(target_ts,min_,max_)
                    #    notscaled = True
                    
                    
                    print(f'new data shape: {data_ts.shape}')
                    if agg_data:
                        data_tot.append(minmax_normalize(data_ts,min_,max_))
                        #data_tot.append(np.transpose(data_ts, (0,2,1)))
                                    

                        theta_tot.append(data_thetas)
  
                print(f'tot theta shape: {np.concatenate(theta_tot, axis=0).shape}')
                print(f'tot data shape: {np.concatenate(data_tot, axis=0).shape}')

                # retrain model after first initial training
                if i > 0:
                    retrain = True
                else:
                    model = Regressor(name_id=f'lv_{i}')
                
                if agg_data:
                    model.train_ts, model.val_ts, model.train_thetas, model.val_thetas = train_test_split(np.concatenate(data_tot, axis=0),
                                                                                                        np.concatenate(theta_tot, axis=0), 
                                                                                                        train_size=0.8, 
                                                                                                        random_state=seed)
                else:
                    model.train_ts, model.val_ts, model.train_thetas, model.val_thetas = train_test_split(data_ts,
                                                                                                        theta[i], 
                                                                                                        train_size=0.8, 
                                                                                                        random_state=seed)
                # start inference
                time_begin = time.time()
                model.run(target=target_ts,num_monte_carlo=num_monte_carlo, batch_size=batch_size, 
                        verbose=False, epochs=epochs, infer_pdf=infer_pdf, retrain=retrain, pooling_len=3)

                
                est_normal = Gaussian(m=model.proposal_mean[0], S=model.proposal_covar[0] + 1e-7)
                if correction:
                    if i > 0:
                        try:
                            est_proposal_prior = est_normal.__div__(est_proposal_prior)
                            est_proposal_prior.S += 1e-7
                        except np.linalg.LinAlgError:
                            print('Improper Gaussian, using previous proposal')
                            pass
                    else:
                        est_proposal_prior = est_normal
                else:
                    est_proposal_prior = est_normal

                samples = est_proposal_prior.gen(Ndata)
                samples_= checkprior(samples)
                theta.append(samples_)
                proposals.append([est_proposal_prior.m, est_proposal_prior.S])
                time_ticks.append(time.time() - time_begin)

            result_posterior.append(theta)
            result_proposal_prior.append(proposals)
            store_time.append(time_ticks)
        return np.array(result_posterior), result_proposal_prior, np.array(store_time)
    except KeyboardInterrupt:        
        return np.asarray(result_posterior), np.asarray(result_proposal_prior), np.asarray(store_time)
    



def bnn_experiment():
    bnn_res = run_bnn(total_runs=5,num_rounds=10,num_data_per_round=2_500,epochs=500,seed=0)
    np.save('bnn_res_theta_5x10x2500.npy',bnn_res_5[0])
    np.save('bnn_res_proposal_5x10x2500.npy',bnn_res_5[1])
    np.save('bnn_res_time_5x10x2500.npy',bnn_res_5[2])

    


