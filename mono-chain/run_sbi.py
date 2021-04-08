#!/usr/bin/env python
# coding: utf-8




import torch
import torch.nn as nn
import torch.nn.functional as F 

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
import numpy as np
import time



from LV_model import loguniform_prior, simulator, lotka_volterra, newData

from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from gillespy2 import VariableSSACSolver



def newData_torch(theta):
    if len(theta.shape)>1:
        theta_ = theta
    else:
        theta_ = np.expand_dims(theta,axis=0)
    #print(theta_.shape)
    data, theta_ = newData(theta_, toexp=True)
    if len(data.shape)>2:
        data = data[0,:,:]
    return(torch.tensor(data))





class lotka_volterra(Model):
    def __init__(self, parameter_values=None):
        Model.__init__(self, name="lotka_volterra")
        self.volume = 1

        ## Parameters
        self.add_parameter(Parameter(name="k1", expression=1.0))
        self.add_parameter(Parameter(name="k2", expression=0.005))
        self.add_parameter(Parameter(name="k3", expression=0.6))

        ## Species
        self.add_species(Species(name='prey', initial_value = 100, mode = 'discrete'))
        self.add_species(Species(name='predator', initial_value = 100, mode = 'discrete'))

        ## Reactions
        self.add_reaction(Reaction(name="r1",
                                   reactants = {'prey' : 1},
                                   products = {'prey' : 2},
                                   rate = self.listOfParameters['k1']))
        self.add_reaction(Reaction(name="r2",
                                   reactants = {'predator': 1, 'prey' : 1},
                                   products = {'predator' : 2},
                                   rate = self.listOfParameters['k2']))
        self.add_reaction(Reaction(name="r3",
                                   reactants = {'predator' : 1},
                                   products = {},
                                   rate = self.listOfParameters['k3']))

        # Timespan
        self.timespan(np.linspace(0, 50, 51))

def simulator(params, model, compiled_solver, toexp = True, transform = True):
    parameter_names = ['k1', 'k2', 'k3']
    if toexp:
        params = np.exp(params)
    
    res = model.run(
            solver = compiled_solver,
            show_labels = True, # remove this
            seed = np.random.randint(1e8), # remove this
            timeout = 3,
            variables = {parameter_names[i] : params[i] for i in range(len(parameter_names))})

    if res.rc == 33:
        nans = np.array([np.nan]).repeat(51)
        return np.array([nans, nans])


    if transform:
        # Extract only observed species
        prey = res['prey']
        predator = res['predator']

        return np.vstack([prey, predator])
    else:
        return res

    
    
def wrapper(param):
    return simulator(param, vilar_model,solver)


vilar_model = lotka_volterra()
solver = VariableSSACSolver(vilar_model)



class SummaryNet_large(nn.Module): 
    
    def __init__(self): 
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=20, kernel_size=3, padding=2)
        # Maxpool layer that reduces YxY image to length (20,4)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=20*5*2, out_features=8) 
        
    def forward(self, x):
        x = x.view(-1, 2, 51)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 20*5*2)
        #x = torch.flatten(x)
        x = F.relu(self.fc(x))
        return x




class SummaryNet(nn.Module): 
    
    def __init__(self): 
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=6, kernel_size=3, padding=2)
        # Maxpool layer that reduces YxY image to length (6,4)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6*5*2, out_features=8) 
        
    def forward(self, x):
        x = x.view(-1, 2, 51)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6*5*2)
        #x = torch.flatten(x)
        x = F.relu(self.fc(x))
        return x






def run_snpe(total_runs=10, num_generation=6, seed=46):
    
    theta_true = np.log([1.0,0.005, 1.0])

    
    torch.manual_seed(seed)
    num_workers = 16 #for parallel execution of simulations
    Ndata = 1000
    use_embedding_net = True
    use_mcmc = False #becomes very slow, but can handle leakage
    result_posterior = []
    store_time = []

    a0, b0 = np.log(0.002), np.log(2)
    prior = utils.BoxUniform(low=torch.tensor([a0,a0,a0]), 
                             high=torch.tensor([b0,b0,b0]))

    simulator_sbi, prior = prepare_for_sbi(wrapper, prior)

    x_o = np.load("target_original_shape_ts.npy")
    x_o = torch.tensor(x_o)
    #x_o = torch.flatten(x_o)#.flatten()reshape((2,51))

    #NN for summary statistic 
    embedding_net = SummaryNet_large()
    
    try:
        for run in range(total_runs):
            print(f"starting run {run}")

            theta_store = []
            time_ticks = []
            posteriors = []
            proposal = prior

            if use_embedding_net:
                neural_posterior = utils.posterior_nn(model='maf', 
                                                    embedding_net=embedding_net,
                                                    hidden_features=10,
                                                    num_transforms=2)
            else:
                neural_posterior = utils.posterior_nn(model='maf', 
                                                    hidden_features=10,
                                                    num_transforms=2)

            inference = SNPE_C(prior=prior, density_estimator=neural_posterior)

            for i in range(num_generation):
                print(f"starting round {i}")
                time_begin = time.time()
                theta, x = simulate_for_sbi(simulator_sbi, proposal, num_simulations=Ndata, num_workers=num_workers)

                mask = torch.tensor(np.invert(np.isnan(x.numpy())[:,0,0]))
                x = x[mask,:,:]
                theta = theta[mask,:]

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
    except KeyboardInterrupt:
        return np.asarray(result_posterior), np.asarray(store_time), posteriors
    return np.asarray(result_posterior), np.asarray(store_time), posteriors



def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        print(name,':', param)
    print(f"Total Trainable Params: {total_params}")
    return total_params





def sbi_experiment():
    posterior, times, snpe_posteriors = run_snpe(total_runs=10, num_generation=10, seed=0)
    np.save('SBI_10_10gen_large.npy', snpe_posteriors)
    np.save('SBI_10_10gen_large_sample.npy',posterior)
    np.save('SBI_10_10gen_large_sample_times.npy',times)
    
    total_count = count_parameters(snpe_posteriors[1].net)
    print(f'total parameter count: {total_count}')

