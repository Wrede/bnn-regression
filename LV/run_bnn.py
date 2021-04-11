#!/usr/bin/env python
# coding: utf-8


from LV_model import uniform_prior, simulator, lotka_volterra, newData,minmax_normalize, checkprior
import logging, dask, scipy, pickle, absl, sklearn
from bcnn_model import Regressor
from sklearn.model_selection import train_test_split 
from gaussian import Gaussian
import numpy as np
import pandas as pd
import time
import copy as cp

from tensorflow.errors import InvalidArgumentError

def run_bnn(total_runs=5, num_rounds=8, num_data_per_round = 5_000, 
            num_monte_carlo=500, batch_size=128,ID='test', seed=None,epochs=100,
            correction=True, agg_data=True, infer_pdf=True, allow_leaks=False):





      
    np.random.seed(seed)
    save_folder = ID
    initial_prior = uniform_prior 
    result_posterior = [] #store posterior samples per round
    result_proposal_prior = [] #store m, S of gaussian proposal
    store_time = [] # time per round without simulation
    try:
        init = initial_prior(left = [0.002, 0.002,0.002],
                             right =[2,2,2])
        for run in range(total_runs):
            print(f'starting run {run}')
            theta = []
            theta_tot = []
            proposals = []
            data_tot = []
            theta_tot = []
            time_ticks = []
            est_proposal_prior = init
            theta.append(est_proposal_prior.gen(num_data_per_round))
            retrain = False
            min_ = None
            max_ = None

            target_ts = np.load('target_ts.npy') #Load the observation
            notscaled = False
            for i in range(num_rounds):
                
                print(f'starting round {i}')

                # generate new data
                if len(theta[i] > 0):
                    data_ts, data_thetas = newData(theta[i], toexp=True)
                    data_ts = np.transpose(data_ts, (0,2,1))
                   
                    
                    print(f'new data shape: {data_ts.shape}')
                    if agg_data:
                        data_ts_,min_,max_ = minmax_normalize(data_ts)
                        data_tot.append(data_ts_)
                        #data_tot.append(np.transpose(data_ts, (0,2,1)))
                                    
                        theta[i] = data_thetas # remove connected to NaN series
                        #theta_tot.append(data_thetas)
                else:
                    print("theta[i] < 0")
                    #raise ValueError("theta[i] < 0")
                
                if not notscaled:
                    target_ts = minmax_normalize(target_ts,min_,max_)[0]
                    notscaled = True
                

                
  
                print(f'tot theta shape: {np.concatenate(theta, axis=0).shape}')
                print(f'tot data shape: {np.concatenate(data_tot, axis=0).shape}')

                # retrain model after first initial training
                if i > 0:
                    retrain = True
                else:
                    model = Regressor(name_id=f'lv_{i}')
                
                if agg_data:
                    model.train_ts, model.val_ts, model.train_thetas, model.val_thetas = train_test_split(np.concatenate(data_tot, axis=0),
                                                                                                        np.concatenate(theta, axis=0), 
                                                                                                        train_size=0.8, 
                                                                                                        random_state=seed)
                else:
                    model.train_ts, model.val_ts, model.train_thetas, model.val_thetas = train_test_split(data_ts,
                                                                                                        theta[i], 
                                                                                                        train_size=0.8, 
                                                                                                        random_state=seed)
                # start inference
                time_begin = time.time()
                model.run(target=target_ts,num_monte_carlo=num_monte_carlo, batch_size=batch_size, verbose=False, epochs=epochs, infer_pdf=infer_pdf, retrain=retrain, pooling_len=3)
                
                est_normal = Gaussian(m=model.proposal_mean[0], S=model.proposal_covar[0] + 1e-7)

                est_proposal_prior_old = cp.copy(est_proposal_prior)
                if correction:
                    if i > 0:
                        try:
                            est_proposal_prior = est_normal.__div__(est_proposal_prior)
                            est_proposal_prior.S += 1e-7
                        except np.linalg.LinAlgError:
                            print('Improper Gaussian, using previous proposal')
                            pass
                        except AssertionError:
                            print('Improper Gaussian, using previous proposal')
                            pass
                    else:
                        est_proposal_prior = est_normal
                else:
                    est_proposal_prior = est_normal

                # take care of prior leakage
                samples = est_proposal_prior.gen(num_data_per_round)
                samples_= checkprior(samples, init)
                if not allow_leaks:
                    if len(samples_) == 0:
                        print('no samples inside of prior, use old prior')
                        est_proposal_prior = est_proposal_prior_old
                        samples = est_proposal_prior.gen(num_data_per_round)
                        samples_= checkprior(samples, init) 
                    while len(samples_) != len(samples):
                        samples_new = est_proposal_prior.gen(num_data_per_round)
                        samples_new = checkprior(samples_new, init)
                        samples_ = np.concatenate([samples_, samples_new], axis=0)
                        if len(samples_) > num_data_per_round:
                            samples_ = samples_[:num_data_per_round,]
                    print('filled leakage')
                    samples = samples_
                else:
                    print("did not remove the leaks")
               
                theta.append(samples)
                proposals.append([est_proposal_prior.m, est_proposal_prior.S])
                time_ticks.append(time.time() - time_begin)

            result_posterior.append(theta)
            result_proposal_prior.append(proposals)
            store_time.append(time_ticks)
        return np.array(result_posterior), result_proposal_prior, np.array(store_time)
    except KeyboardInterrupt:
        print("keyboard stop")
        return np.array(result_posterior), result_proposal_prior, np.array(store_time)
    except InvalidArgumentError:
        print("******* stopped early! ********")
        return np.array(result_posterior), result_proposal_prior, np.array(store_time)

    



def bnn_experiment():
    bnn_res = run_bnn(total_runs=5,num_rounds=8,num_data_per_round=5_000,epochs=500,seed=1,
                     allow_leaks=False)
    np.save('bnn_res_theta_5x8x5000.npy',bnn_res[0])
    np.save('bnn_res_proposal_5x8x5000.npy',bnn_res[1])
    np.save('bnn_res_time_5x8x5000.npy',bnn_res[2])

    


