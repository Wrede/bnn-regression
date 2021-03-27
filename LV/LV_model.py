from scipy.stats import loguniform

def loguniform_prior(Ndata=2_500, log=True):
    a0, b0 = 0.002, 2
    a1, b1 = 0.002, 2
    k1 = loguniform.rvs(a0,b0,size=Ndata)
    k2 = loguniform.rvs(a1,b1,size=Ndata)
    k3 = loguniform.rvs(a0,b0,size=Ndata)
    theta = np.vstack((k1,k2,k3)).T
    if log:
        return np.log(theta)
    return theta

import numpy as np
from dask import compute
from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from gillespy2 import VariableSSACSolver

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
        return np.nan


    if transform:
        # Extract only observed species
        prey = res['prey']
        predator = res['predator']

        return np.vstack([prey, predator])[np.newaxis,:,:]
    else:
        return res


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

def newData(new_thetas,toexp=False):
    model = lotka_volterra()
    compiled_solver = VariableSSACSolver(model)

    new_data = [simulator(x, model, compiled_solver, toexp=toexp) for x in new_thetas]
    new_data, = compute(new_data)
    new_data, = compute(new_data)

    x = []
    y = []
    for e,i in enumerate(new_data):
        if type(i) == float:
            continue
        elif np.max(i) > 2000:
            continue
        else:
            x.append(i)
            y.append(new_thetas[e])

    new_data = np.asarray(x)
    new_thetas = np.asarray(y)
    new_data = np.reshape(new_data, (new_data.shape[0],new_data.shape[2], new_data.shape[3]))
    return (new_data, new_thetas)