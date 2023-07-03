from abc import abstractmethod

import numpy as np
import scipy
import math
import torch

import matplotlib.pyplot as plt

class NonconformityScore():
    
    def __init__(self, model, verbose = False):
        self.model = model
        self.verbose = verbose
    
    def __call__(self, X, y):
        pass
    
    def quantile(self, X, y, alpha, inflate = True):
        scores = self(X, y)
        scores, _ = torch.sort(scores)
        level = min((1 + ((1 / scores.shape[0]) if inflate else 0)) * (1 - alpha), 1)
        index = math.ceil(level * scores.shape[0]) - 1
        return scores[index].item()

    def predict(self, X):
        pass

class SwissKnife(NonconformityScore):

    def __init__(self, model, verbose = False):
        super().__init__(model, verbose)
        
        self.preds = None
        self.pi = None

        self.scores = {}
        self.hash = 0

    def __call__(self, X, y, mode = "MSE"):

        self.predict(X)

        self.scores["MSE"] = torch.abs(self.preds[:, 0] - y)
        self.scores["residual"] = self.scores["res"] = self.preds[:, 0] - y
        self.scores["nMSE"] = self.scores["std"] = torch.abs(self.preds[:, 0] - y) / torch.sqrt(self.preds[:, 1])
        self.scores["variance"] = torch.abs(self.preds[:, 0] - y) / self.preds[:, 1]
        self.scores["interval"] = torch.maximum(self.pi[:, 0] - y, y - self.pi[:, 1])

        if mode is not None:
            return self.scores[mode]

    def quantile(self, X, y, alpha, inflate = True, mode = "MSE"):
        scores = self(X, y, mode)
        scores, _ = torch.sort(scores)
        level = min((1 + ((1 / scores.shape[0]) if inflate else 0)) * (1 - alpha), 1)
        index = math.ceil(level * scores.shape[0]) - 1
        return scores[index].item()

    def predict(self, X):

        if hash(X) != self.hash:

            self.results = {}
            self.hash = hash(X)

            self.preds, self.pi = self.model(X)

class CachedSwissKnife(NonconformityScore):

    def __init__(self, model, X, y, verbose = False):
        super().__init__(model, verbose)

        self.y = y
        self.predict(X)

        self.scores = {}
        self.scores["MSE"] = torch.abs(self.preds[:, 0] - y)
        self.scores["residual"] = self.scores["res"] = self.preds[:, 0] - y
        self.scores["nMSE"] = self.scores["std"] = torch.abs(self.preds[:, 0] - y) / torch.sqrt(self.preds[:, 1])
        self.scores["variance"] = torch.abs(self.preds[:, 0] - y) / self.preds[:, 1]
        self.scores["interval"] = torch.maximum(self.pi[:, 0] - y, y - self.pi[:, 1])

    def __call__(self, index, mode = "MSE"):
        
        if mode is not None:
            return self.scores[mode][index]

    def quantile(self, index, alpha, inflate = True, mode = "MSE"):
        scores = self(index, mode)
        scores, _ = torch.sort(scores)
        level = min((1 + ((1 / scores.shape[0]) if inflate else 0)) * (1 - alpha), 1)
        index = math.ceil(level * scores.shape[0]) - 1
        return scores[index].item()

    def predict(self, X):
        self.preds, self.pi = self.model(X)


class MSEScore(NonconformityScore):
    
    def __call__(self, X, y):
        preds = self.model(X)
        scores = torch.abs(preds - y)
        return scores

class ResidualScore(NonconformityScore):
    
    def __call__(self, X, y):
        preds = self.model(X)
        scores = preds - y
        return scores

class NormalizedMSEScore(NonconformityScore):
    
    def __call__(self, X, y):
        preds = self.model(X)
        scores = torch.abs(preds[:, 0] - y) / preds[:, 1]
        return scores
    
class IntervalScore(NonconformityScore):
    
    def __call__(self, X, y):
        preds = self.model(X)
        scores = torch.maximum(preds[:, 0] - y, y - preds[:, 1])
        return scores

class ICP():

    def __init__(self, measure: NonconformityScore):
        self.measure = measure

    @abstractmethod
    def conformalize(self, X_cal, y_cal, preds):
        pass

class NCP(ICP):

    def conformalize(self, X_cal, y_cal, preds):
        score = self.measure(X_cal, y_cal)
        return torch.stack([preds[:, 0] - score * preds[:, 1], preds[:, 0] + score * preds[:, 1]], dim = -1)

class UniformICP(ICP):

    def conformalize(self, X_cal, y_cal, preds):
        score = self.measure(X_cal, y_cal)
        return torch.stack([preds - score, preds + score], dim = -1)











    
def prob_nc(f, x, y):
    prob = f(x)
    return 1 - np.array([prob[i, y[i]] for i in range(x.shape[0])])

def binary_prob_nc(f, x, y):
    prob = f(x)
    return np.array([(1 - prob[i] if y[i] else prob[i]) for i in range(x.shape[0])])

def point_nc(f, x, y):
    return np.abs(f(x) - y)

def interval_nc(f, x, y):
    preds = f(x)
    return np.maximum(preds[:, 0] - y, y - preds[:, 1])

def nesting_nc(f, x, y):
    preds = f(x)
    return np.argmin((preds[:, 0] <= y) & (preds[:, 1] >= y), axis = 1)

def distr_nc(f, x, y, window = 100):
    from sklearn.neighbors import KernelDensity

    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)

    dens1 = KernelDensity(kernel = "gaussian", bandwidth = 0.1).fit(y[:min(window, x.shape[0]), :])
    dens2 = KernelDensity(kernel = "gaussian", bandwidth = 0.1).fit(y[max(-window, -x.shape[0]):, :])
    return dens1.score_samples(y) - dens2.score_samples(y)

def distr_nc_factory(window):
    return lambda x, y, z: distr_nc(x, y, z, window)

def threshold(scores, alpha, inflate = True, safe = True):
    if scores.shape[0] == 0:
        if safe:
            return np.nan
        else:
            raise Exception("Scores has length 0.")
    level = min((1 + ((1 / scores.shape[0]) if inflate else 0)) * (1 - alpha), 1)
    index = math.ceil(level * scores.shape[0]) - 1
    return np.sort(scores)[index]

def error(y, lower, upper):
    return np.mean((lower > y) | (upper < y))

def online_point_conformalize(f, x_val, y_val, x, y, alpha = 0.1, block_size = 100, inflate = True, verbose = False):
    if verbose:
        from tqdm import tqdm

    x_all = np.concatenate([x_val, x])
    y_all = np.concatenate([y_val, y])
    preds = f(x)

    bounds = np.zeros((x.shape[0], 2))
    for i in tqdm(range(x.shape[0] // block_size + 1)) if verbose else range(x.shape[0] // block_size + 1):
        scores = point_nc(f, x_all[:x_val.shape[0] +\
                             min(i * block_size, x.shape[0]), :], y_all[:x_val.shape[0] + min(i * block_size, x.shape[0])])

        if(len(scores) != len(np.unique(scores))):
            print("ties detected:", len(scores) - len(np.unique(scores)))

        crit = threshold(scores, alpha, inflate)
        bounds[max(0, i - 1) * block_size:min((i + 1) * block_size, x.shape[0]), 0] = preds[max(0, i - 1) * block_size:min((i + 1) * block_size, x.shape[0])] - crit
        bounds[max(0, i - 1) * block_size:min((i + 1) * block_size, x.shape[0]), 1] = preds[max(0, i - 1) * block_size:min((i + 1) * block_size, x.shape[0])] + crit

    return bounds

def inductive_pvalues(f, nc, x, y, x_val, y_val, verbose = False):
    if verbose:
        from tqdm import tqdm

    scores = nc(f, x, y)
    calibration = nc(f, x_val, y_val)
    if verbose and (len(calibration) != len(np.unique(calibration))):
        print("ties detected:", len(calibration) - len(np.unique(calibration)))

    p = np.zeros(x.shape[0])
    for i in tqdm(range(x.shape[0])) if verbose else range(x.shape[0]):
        p[i] = (np.sum(scores[i] <= calibration) + 1) / (scores.shape[0] + 1)
    return p

def online_pvalues(f, nc, x, y, x_val = None, y_val = None, verbose = False):
    if verbose:
        from tqdm import tqdm

    scores = nc(f, x, y)
    calibration = np.concatenate([nc(f, x_val, y_val), scores]) if x_val else scores
    if verbose and (len(calibration) != len(np.unique(calibration))):
        print("ties detected:", len(calibration) - len(np.unique(calibration)))

    p = np.zeros(x.shape[0])
    for i in tqdm(range(x.shape[0])) if verbose else range(x.shape[0]):
        p[i] = (np.sum(scores[i] <= calibration[:-scores.shape[0]+i]) + 1) / (calibration[:-scores.shape[0]+i].shape[0] + 1)
    return p

def sequential_pvalues(f, nc, x, y, verbose = False):
    if verbose:
        from tqdm import tqdm

    scores = nc(f, x, y)
    if verbose and (len(scores) != len(np.unique(scores))):
        print("ties detected:", len(scores) - len(np.unique(scores)))

    p = np.zeros(x.shape[0])
    for i in tqdm(range(x.shape[0])) if verbose else range(x.shape[0]):
        p[i] = np.mean(scores[i] <= scores[:i+1])
    return p

def simple_mixture_martingale(p, mesh_size = 0.0001, log_scale = True):
    mesh = np.arange(0, 1 + mesh_size, mesh_size)
    integrals = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        if i == 0:
            martingale = np.array([(x * p[0] ** (x - 1)) for x in mesh])
        else:
            martingale *= np.array([(x * p[i] ** (x - 1)) for x in mesh])
        integrals[i] = np.sum((martingale[:-1] + martingale[1:])) * (mesh_size / 2)
    return np.log(integrals) if log_scale else integrals

def kde_martingale(p):
    from sklearn.neighbors import KernelDensity

    martingale = np.zeros(p.shape[0])
    p = np.expand_dims(p, 1)
    for i in range(1, p.shape[0]):
        kde = KernelDensity(kernel = "gaussian").fit(np.concatenate([p[:i, :], -p[:i, :], 2 - p[:i, :]]))
        integral, _ = scipy.integrate.quad(lambda x: np.exp(kde.score_samples([[x]])), 0, 1)
        martingale[i] = martingale[i - 1] + np.log(np.exp(kde.score_samples([p[i]])) / integral)
    return martingale

def histogram(x, range = None, bins = 20):
    hist, edges = np.histogram(x, range = range, bins = bins)
    hist += bins
    normalization = np.sum(hist)
    return lambda y: np.mean(hist[(edges[:-1] <= y) & (edges[1:] >= y)]) / normalization

def histogram_martingale(p, bins = 20, verbose = False):
    if verbose:
        from tqdm import tqdm

    for i in tqdm(range(p.shape[0])) if verbose else range(p.shape[0]):
        if i == 0:
            martingale = np.zeros(p.shape[0])
        else:
            hist = histogram(p[:i], bins = bins, range = [0, 1])
            martingale[i] = martingale[i - 1] + np.log(bins * hist(p[i]))
    return martingale
