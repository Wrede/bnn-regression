"""
The Moving Averages 2 model
"""
from dask import delayed, compute
import numpy as np

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

    return y

def triangle_prior(n=10):
    p = []
    trials = 0
    acc = 0
    while acc < n:
        trials += 1
        r = np.random.rand(2) * np.array([4, 2]) + np.array([-2, -1])
        if r[1] + r[0] >= -1 and r[1] - r[0] >= -1:
            p.append(r)
            acc += 1
    return np.asarray(p)

def uniform_prior(n):
    _min = np.array([-2,-1])
    _max = np.array([4, 2])
    samples = np.empty((n,2))
    
    for i in range(n):
        r = np.random.rand(2) * _max + _min
        samples[i] = r
    return samples
    

def newData(new_thetas):
    new_data = [delayed(simulator)(x) for x in new_thetas]
    new_data, = compute(new_data)
    
    new_data = np.asarray(new_data)
    new_thetas = np.asarray(new_thetas)
    new_data = np.reshape(new_data, (new_data.shape[0],new_data.shape[-1],1))
    return (new_data, new_thetas)
