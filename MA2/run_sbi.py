import torch
import torch.nn as nn
import torch.nn.functional as F 

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
import numpy as np
import time
import os

class SummaryNet(nn.Module): 
    
    def __init__(self): 
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6*10, out_features=8) 
        
    def forward(self, x):
        x = x.view(-1, 1, 100)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6*10)
        x = F.relu(self.fc(x))
        return x

def simulator(param, n=100):
    """
    Simulate a given parameter combination.

    Parameters
    ----------
    param : vector or 1D array
        Parameters to simulate (\theta).
    n : integer
        Time series length
    """
    m = len(param)
    g = np.random.normal(0, 1, n)
    gy = np.random.normal(0, 0.3, n)
    y = np.zeros(n)
    x = np.zeros(n)
    for t in range(0, n):
        x[t] += g[t]
        for p in range(0, np.minimum(t, m)):
            x[t] += g[t - 1 - p] * param[p]
        y[t] = x[t] + gy[t]
    
    y = torch.tensor(y)
    return y

def run_snpe(total_runs=10, num_generation=6, seed=46, nde='maf'):
    torch.manual_seed(seed)
    num_workers = 16 #for parallel execution of simulations
    Ndata = 3000
    use_embedding_net = True
    use_mcmc = False #becomes very slow, but can handle leakage
    result_posterior = []
    store_time = []

    prior = utils.BoxUniform(low=torch.tensor([-2.0,-1.0]), 
                         high=torch.tensor([2.0,1.0]))

    simulator_sbi, prior = prepare_for_sbi(simulator, prior)
    
    x_o = np.load("target_ts.npy")
    x_o = torch.tensor(x_o)
    x_o = x_o.reshape((1,100))

    #NN for summary statistic 
    embedding_net = SummaryNet()
    
    for run in range(total_runs):
        print(f"staring run {run}")

        theta_store = []
        time_ticks = []
        posteriors = []
        proposal = prior

        if use_embedding_net:
            neural_posterior = utils.posterior_nn(model=nde, 
                                                embedding_net=embedding_net,
                                                hidden_features=10,
                                                num_transforms=2)
        else:
            neural_posterior = utils.posterior_nn(model=nde, 
                                                hidden_features=10,
                                                num_transforms=2)
        
        inference = SNPE_C(prior=prior, density_estimator=neural_posterior)
    
        for i in range(num_generation):
            print(f"staring round {i}")
            time_begin = time.time()
            theta, x = simulate_for_sbi(simulator_sbi, proposal, num_simulations=Ndata, num_workers=num_workers)
            
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator, sample_with_mcmc=use_mcmc)
            print("building post done")
            posteriors.append(posterior)
            proposal = posterior.set_default_x(x_o)

            posterior_samples = posterior.sample((Ndata,), x=x_o).numpy()
            print("Post samples done")
            theta_store.append(posterior_samples)
            time_ticks.append(time.time() - time_begin)
        
        result_posterior.append(theta_store)
        store_time.append(time_ticks)
    return np.asarray(result_posterior), np.asarray(store_time), posteriors

def sbi_experiment():
    ID = 'data'
    try:
        os.mkdir(ID)
    except FileExistsError:
        print(f'{ID} folder already exists, continue...')

    sbi_post, sbi_time, sbi_post_object = run_snpe(total_runs=10, num_generation=6, seed=2, nde='maf')
    np.save(f'{ID}/sbi_{ID}_post', sbi_post)
    np.save(f'{ID}/sbi_{ID}_time', sbi_time)