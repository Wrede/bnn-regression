#!/usr/bin/env python
# coding: utf-8


import sciope



import numpy as np
import gillespy2
from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from gillespy2 import EventAssignment, EventTrigger, Event

from sciope.utilities.priors.uniform_prior import UniformPrior
from sciope.utilities.summarystats.identity import Identity

import matplotlib.pyplot as plt

# To run a simulation using the SSA Solver simply omit the solver argument from model.run().
from gillespy2 import VariableSSACSolver
# from gillespy2 import TauLeapingSolver
# from gillespy2 import TauHybridSolver
# from gillespy2 import ODESolver

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


class lotka_volterra(Model):
    def __init__(self, parameter_values=None):
        Model.__init__(self, name="lotka_volterra")
        self.volume = 1

        # Parameters
        self.add_parameter(Parameter(name="k1", expression=1.0))
        self.add_parameter(Parameter(name="k2", expression=0.005))
        self.add_parameter(Parameter(name="k3", expression=1.0))

        # Species
        self.add_species(Species(name='prey', initial_value = 100, mode = 'discrete'))
        self.add_species(Species(name='predator', initial_value = 100, mode = 'discrete'))
        
        # Reactions
        self.add_reaction(Reaction(name="r1", reactants = {'prey' : 1}, products = {'prey' : 2}, rate = self.listOfParameters['k1']))
        self.add_reaction(Reaction(name="r2", reactants = {'predator' : 1, 'prey' : 1}, products = {'predator' : 2}, rate = self.listOfParameters['k2']))
        self.add_reaction(Reaction(name="r3", reactants = {'predator' : 1}, products = {}, rate = self.listOfParameters['k3']))

        # Timespan
        self.timespan(np.linspace(0, 50, 51))
        
model = lotka_volterra()
compiled_solver = VariableSSACSolver(model)


# # Data



#target_ts = np.load('target_ts.npy')
obs_data = np.load('target_original_shape_ts.npy')


# # Prior Distributions




theta_true = [0,-5.8,0]





parameter_names = ['k1', 'k2', 'k3']
a0, b0 = np.log(0.002), np.log(2)
lower_bounds_wide = [a0, a0, a0]
upper_bounds_wide = [b0, b0, b0]
prior_wide = UniformPrior(np.array(lower_bounds_wide), np.array(upper_bounds_wide))


# # Simulator


# Here we use the GillesPy2 Solver
from dask import delayed, compute

def simulator(params, model, transform = True):    
    res = model.run(
            solver = compiled_solver,
            show_labels = True, # remove this
            seed = np.random.randint(1e8), # remove this
            timeout = 3,
            variables = {parameter_names[i] : np.exp(params[i]) for i in range(len(parameter_names))})
    
    if res.rc == 33:
        return np.inf * np.ones((1,2,51))
    
    
    if transform:
        # Extract only observed species
        prey = res['prey']
        predator = res['predator']

        return np.vstack([prey, predator])[np.newaxis,:,:]
    else:
        return res

# Wrapper, simulator function to abc should should only take one argument (the parameter point)
def simulator2(x, transform = True):
    return simulator(x, model=model, transform = transform)


# # Summary Statistics and Distance Function

# ### Identity Statistic and Euclidean Distance


from sciope.utilities.summarystats.identity import Identity
from sciope.utilities.distancefunctions.euclidean import EuclideanDistance

normalization_values = np.max(obs_data, axis = 2)[0,:]
def max_normalization(data, norm_val=normalization_values):
    dc = data[0].reshape(1,2,51).copy().astype(np.float32)
    dc_ = np.array(dc, copy=True)
    dc_[:,0,:] = dc[:,0,:]/norm_val[0]
    dc_[:,1,:] = dc[:,1,:]/norm_val[1]
    return dc_

summary_stat = Identity(max_normalization)
distance_func = EuclideanDistance()


# # Inference

# ### Using ABC-SMC



from sciope.inference.smc_abc import SMCABC
from sciope.utilities.perturbationkernels.multivariate_normal import MultivariateNormalKernel
from sciope.utilities.epsilonselectors.relative_epsilon_selector import RelativeEpsilonSelector

dim = prior.get_dimension()
pk = MultivariateNormalKernel(
                d=dim,
                adapt=False, cov=0.05 * np.eye(dim))

maximum_number_of_rounds = 8
eps_selector = RelativeEpsilonSelector(20, maximum_number_of_rounds)

smcabc = SMCABC(obs_data, # Observed Dataset
          simulator2, # Simulator method
          prior_wide, # Prior
          summaries_function=summary_stat.compute,
          perturbation_kernel = pk,
          use_logger = False
          )



import time

np.random.seed(0)
max_gen = 5
smc_abc_gen = []
time_ticks = []
res_gen = []
for i in range(max_gen):
    time_begin = time.time()
    smc_abc_results = smcabc.infer(num_samples = 1000, batch_size = 1000, chunk_size=1, eps_selector=eps_selector)
    time_ticks.append(time.time() - time_begin)

    res_gen.append(smc_abc_results)
    posterior = np.vstack(smc_abc_results[-1]['accepted_samples'])
    gen_post = np.array([x['accepted_samples'] for x in smc_abc_results])
    smc_abc_gen.append(gen_post)
np.save('smcabc_posterior_5gen.npy',smc_abc_gen)
np.save('smcabc_posterior_5gen_time.npy',time_ticks)
np.save('smcabc_posterior_5gen_res.npy',res_gen)



tot = []
for i in range(len(smc_abc_results)):
    tot.append(smc_abc_results[i]['trial_count'])
print(f'total number of simulations used {np.cumsum(tot)}')

