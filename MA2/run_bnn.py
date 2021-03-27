from bcnn_model import Regressor
from ma2_model import newData, triangle_prior, uniform_prior
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as rng
import scipy.stats
import time
import os

class Gaussian():
    def __init__(self, m=None, U=None, S=None, P=None, Pm=None):
        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        else:
            raise ValueError('Mean information missing.')
    def gen(self, n_samples=1):
        """Returns independent samples from the gaussian."""

        z = rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m

        return samples
    
    def eval(self, x, ii=None, log=True):
        """
        Evaluates the gaussian pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S)
            lp = np.array([lp]) if x.shape[0] == 1 else lp

        res = lp if log else np.exp(lp)
        return res
    
    def __mul__(self, other):
        """Multiply with another gaussian."""

        assert isinstance(other, Gaussian)

        P = self.P + other.P
        Pm = self.Pm + other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another gaussian."""

        assert isinstance(other, Gaussian)

        res = self * other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __div__(self, other):
        """Divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, Gaussian)

        P = self.P - other.P
        Pm = self.Pm - other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __idiv__(self, other):
        """Incrementally divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, Gaussian)

        res = self / other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __pow__(self, power, modulo=None):
        """Raise gaussian to a power and get another gaussian."""

        P = power * self.P
        Pm = power * self.Pm

        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise gaussian to a power."""

        res = self ** power

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another gaussian, i.e. KL(this | other)."""

        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim

        t1 = np.sum(other.P * self.S)

        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))

        t3 = self.logdetP - other.logdetP

        t = 0.5 * (t1 + t2 + t3 - self.ndim)

        return t


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
    #initial_prior = uniform_prior 
    initial_prior = triangle_prior
    result_posterior = [] #store posterior samples per round
    result_proposal_prior = [] #store m, S of gaussian proposal
    store_time = [] # time per round without simulation

    target_ts = np.load('target_ts.npy') #Load the observation

    try:

        for run in range(total_runs):
            print(f'starting run {run}')
            theta = []
            proposals = []
            data_tot = []
            time_ticks = []
            theta.append(initial_prior(Ndata))
            retrain = False

            for i in range(num_rounds):
                
                print(f'starting round {i}')

                # generate new data
                data_ts, data_thetas = newData(theta[i])
                if agg_data:
                    data_tot.append(data_ts)

                # retrain model after first initial training
                if i > 0:
                    retrain = True
                else:
                    model = Regressor(name_id=f'ma2_{i}')
                
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
                model.run(target=target_ts,num_monte_carlo=num_monte_carlo, batch_size=batch_size, 
                        verbose=False, epochs=epochs, infer_pdf=infer_pdf, retrain=retrain)

                # save model for evaluation
                #if multi_dim:
                #    lv_c.model1.save(f'{save_folder}/ma2_run{run}_gen{i}_model_multidim_{ID}')
                #else:
                #    lv_c.model1.save(f'{save_folder}/ma2_run{run}_gen{i}_model1_{ID}')
                #    lv_c.model2.save(f'{save_folder}/ma2_run{run}_gen{i}_model2_{ID}')


                #
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
                theta.append(samples)
                proposals.append([est_proposal_prior.m, est_proposal_prior.S])
                time_ticks.append(time.time() - time_begin)

            result_posterior.append(theta)
            result_proposal_prior.append(proposals)
            store_time.append(time_ticks)
        return np.array(result_posterior), result_proposal_prior, np.array(store_time)
    except KeyboardInterrupt:
        return np.asarray(result_posterior), np.asarray(result_proposal_prior), np.asarray(store_time)


def bnn_experiment():
    ID = 'data'
    try:
        os.mkdir(ID)
    except FileExistsError:
        print(f'{ID} folder already exists, continue...')

    #using correction
    bcnn_post, bcnn_proposals, bcnn_time = run_bnn(total_runs=10, num_rounds=6, seed=3, ID=ID)
    np.save(f'{ID}/bcnn_{ID}_correction_samples', bcnn_post)
    #np.save(f'{ID}/bcnn_{ID}_correction_proposal_pdf', bcnn_proposals)
    np.save(f'{ID}/bcnn_{ID}_correction_time', bcnn_time)

    #without correction   
    bcnn_post, bcnn_proposals, bcnn_time = run_bnn(total_runs=10, num_rounds=6, seed=3, ID=ID, correction=False)
    np.save(f'{ID}/bcnn_{ID}_samples', bcnn_post)
    #np.save(f'{ID}/bcnn_{ID}_proposal_pdf', bcnn_proposals)
    np.save(f'{ID}/bcnn_{ID}_time', bcnn_time)





        


