#!/usr/bin/env python
# coding: utf-8

from LV_model import loguniform_prior, simulator, lotka_volterra, newData
from bnn_model import LVClassifier, sample_local, sample_local_2D, inBin
import numpy as np
import pandas as pd
import time




def run_bnn(max_rounds=10,max_gen=10,Ndata=1000,seed=0, multi_dim=False, num_bins=10, thresh=0.0, verbose=False):
    """
    Run the BNN for multiple rounds and multiple generations
    
    *param*
    max_rounds: the number of rounds to run, i.e., new seeds.
    max_gen:    the number of sequential, adaptive, generations.
    Ndata:      max number of model simulations per generation.
    seed:       random number seed
    multi_dim:  solve marginal or cross-terms
    num_bins:   binning of data
    thresh:     cut-of when resampling.
    """
    
    np.random.seed(seed)


    use_small = True # use smaller network arch.
    samplePrior = loguniform_prior
    
    

    target_ts = np.load('target_ts.npy')
    
    res_per_round = {'theta': [], 'theta_corrected': [], 'time': []}
    for round_ in range(max_rounds):
        print(f'round {round_+1} out of {max_rounds}')

        theta = []
        theta_corrected = []
        theta.append(samplePrior(Ndata, True))
        time_ticks = []

        for i in range(max_gen):
            print(f'gen {i+1} out of {max_gen}')            
            time_begin = time.time()

            # of the previous gen dataset, which ones can we re-use? Goal is to maximize 
            # the number of datapoints available.
            if i > 0:
                data_ts_, data_thetas_ = inBin(data_ts, data_thetas, theta[i])

            # generate new data
            data_ts, data_thetas = newData(theta[i], toexp=True)

            if i > 0:
                data_ts = np.append(data_ts, data_ts_, axis=0)
                data_thetas = np.append(data_thetas, data_thetas_, axis=0)

            # saving not only the full parameter arrays THETA but also the ones that are 
            # removed because of timeout signal. 
            theta_corrected.append(data_thetas)

            # Classify new data
            lv_c = LVClassifier(name_id=f'lv_{i}', seed=0)

            lv_c.train_thetas = data_thetas
            lv_c.train_ts = lv_c.reshape_data(data_ts)

            #if multi_dim:
            #    num_bins = 5
            #else:
            #    num_bins = 10
                
            lv_c.run(target=target_ts, num_bins=num_bins, batch_size=128, split=True, toload=False,
                     verbose=verbose, use_small=use_small, multi_dim=multi_dim)

            # save model for evaluation
            #lv_c.model1.save(f'lv_gen{i}_model1')
            #lv_c.model2.save(f'lv_gen{i}_model2')
            #lv_c.model3.save(f'lv_gen{i}_model3')

            # resample
            if multi_dim:
                new_rate1, new_bins1 = sample_local_2D(lv_c.probs1, lv_c.multidim_bins_rate12, num_samples=Ndata, use_thresh=True, thresh=thresh)
                new_rate2, new_bins2 = sample_local_2D(lv_c.probs2, lv_c.multidim_bins_rate13, num_samples=Ndata, use_thresh=True, thresh=thresh)
                new_rate3, new_bins3 = sample_local_2D(lv_c.probs3, lv_c.multidim_bins_rate23, num_samples=Ndata, use_thresh=True, thresh=thresh)
            
                rate1 = np.hstack([new_rate1[:,0], new_rate2[:,0]])
                rate1_ = np.random.choice(rate1, Ndata) # we only need Ndata samples.
                rate2 = np.hstack([new_rate1[:,1], new_rate3[:,0]])
                rate2_ = np.random.choice(rate2, Ndata)
                rate3 = np.hstack([new_rate2[:,1], new_rate3[:,1]])
                rate3_ = np.random.choice(rate3, Ndata)
                new_rate_all = np.vstack([rate1_,rate2_,rate3_]).T
            else:
                new_rate1, new_bins1 = sample_local(lv_c.probs1, lv_c.bins_rate1, num_samples=Ndata, use_thresh=True, thresh=thresh)
                new_rate2, new_bins2 = sample_local(lv_c.probs2, lv_c.bins_rate2, num_samples=Ndata, use_thresh=True, thresh=thresh)
                new_rate3, new_bins3 = sample_local(lv_c.probs3, lv_c.bins_rate3, num_samples=Ndata, use_thresh=True, thresh=thresh)
                new_rate_all = np.vstack((new_rate1,new_rate2,new_rate3)).T
            theta.append(new_rate_all)
           
        
            time_ticks.append(time.time() - time_begin)
            
        res_per_round['theta'].append(theta)
        res_per_round['theta_corrected'].append(theta_corrected)
        res_per_round['time'].append(time_ticks)

    print('*** DONE ***')
    return res_per_round




# test the effect of bins
def bnn_experiment():
    bnn_res_5 = run_bnn(max_rounds=5,max_gen=8,Ndata=1000,seed=0, multi_dim=True, num_bins=5,thresh=0.05)
    np.save('bnn_res_5_5round_8gen_theta_thresh.npy',bnn_res_5['theta'])
    np.save('bnn_res_5_5round_8gen_time_thresh.npy',bnn_res_5['time'])

    bnn_res_4 = run_bnn(max_rounds=5,max_gen=8,Ndata=1000,seed=0, multi_dim=True, num_bins=4,thresh=0.05)
    np.save('bnn_res_4_5round_8gen_theta_thresh.npy',bnn_res_4['theta'])
    np.save('bnn_res_4_5round_8gen_time_thresh.npy',bnn_res_4['time'])

    bnn_res_3 = run_bnn(max_rounds=5,max_gen=8,Ndata=1000,seed=0, multi_dim=True, num_bins=3,thresh=0.05)
    np.save('bnn_res_3_5round_8gen_theta_thresh.npy',bnn_res_3['theta'])
    np.save('bnn_res_3_5round_8gen_time_thresh.npy',bnn_res_3['time'])

    bnn_res_5_not = run_bnn(max_rounds=5,max_gen=8,Ndata=1000,seed=0, multi_dim=True, num_bins=5,thresh=0.0)
    np.save('bnn_res_5_5round_8gen_theta.npy',bnn_res_5_not['theta'])
    np.save('bnn_res_5_5round_8gen_time.npy',bnn_res_5_not['time'])


