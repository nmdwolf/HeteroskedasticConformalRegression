import numpy as np

import torch
import torch.nn as nn

from tqdm.notebook import tqdm

DROP = 0.1
DIM = 64
RATE = 5e-4
DECAY = 1e-6
ENSEMBLE = 50
EPOCHS = 100
LAYERS = 1

# removes crossing incidents
def crossing_act(l):

    a = l[..., 0]
    b = a + nn.ReLU(l[..., 1] - a)

    return torch.stack([a, b], axis = 1)

def conditional_UR_1d(f, X, grid_size = 1000, min = -100, max = 100):
    sample = np.zeros(X.shape[0])
    a = np.zeros_like(sample)
    b = np.zeros_like(a)

    done = np.zeros_like(sample).astype(bool)
    for i in range(min, max):
        index = (f(np.full(X.shape[0], i), X) > 0) * (~done)
        a[index] = i-1
        done[index] = True

    done = np.zeros_like(sample).astype(bool)
    for i in range(max, min, -1):
        index = (f(np.full(X.shape[0], i), X) > 0) * (~done)
        b[index] = i+1
        done[index] = True

    q0 = -np.ones_like(sample)
    grid = np.linspace(a, b, grid_size)
    for i in range(grid_size):
        q0 = np.maximum(q0, f(grid[i, :], X))

    todo = np.ones_like(sample).astype(bool)
    while np.sum(todo) > 0:
        y = np.random.uniform(a, b)
        q = np.random.uniform(0, q0[todo])

        temp = f(y[todo], X[todo, :]) >= q
        index = np.linspace(0, X.shape[0]-1, X.shape[0])[todo][temp].astype(int)
        sample[index] = y[index]

        todo[todo] = ~temp

    return sample
        

def MHMC_1d(f, size, burnin = 500, grid_size = .1):

    sample = np.zeros(size)

    grid = np.arange(-50, 50, grid_size)
    mean = np.sum([x * f(x) for x in grid]) * grid_size
    std = np.mean([(x - mean) ** 2 * f(x) for x in grid]) * grid_size

    x0 = 0
    while f(x0) <= 0:
        x0 = np.random.normal()

    for i in range(size + burnin):
        x = x0 + np.random.normal(mean, std)
        q = f(x) / f(x0)
        if q >= 1 or q >= np.random.uniform(0, 1):
            x0 = x

        if i >= burnin:
            sample[i - burnin] = x0


class Sampler():

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self, X, y):
        pass

class MinibatchSampler(Sampler):

    def __init__(self, batch_size = 256, seed = 15):
        super().__init__(batch_size)
        self.seed = seed

    def sample(self, X, y):
        shuffle_idx = np.arange(X.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(shuffle_idx)

        batch = np.zeros((X.shape[0] // self.batch_size, self.batch_size), dtype = "int")
        weights = np.ones(X.shape[0])

        for i in range(batch.shape[0]):
            batch[i, :] = shuffle_idx[i * self.batch_size:(i+1) * self.batch_size]

        return batch, weights

class HTSampler(Sampler):

    # parameter mode:
    #   * "feature": Calculates distance in feature space
    #   * "target": Calculates distance in target space
    #   * "combi": Calculates distance in product of feature and target space
    def __init__(self, clusters = 3, neighbour_samples = 40, neighbours = 60, \
        batches = None, mode = "feature"):
        super().__init__(clusters * neighbour_samples)
        self.clusters = clusters
        self.neighbours = neighbours
        self.batches = batches
        self.mode = mode

    def sample(self, X, y):

        size = X.shape[0]

        neighbour_index = np.zeros((size, self.neighbours), dtype = "int")
        for i in range(size):
            if self.mode == "feature":
                distance = np.linalg.norm(X - X[i, :], axis = 1)
            elif self.mode == "target":
                if len(y.shape) == 1:
                    y = y[:, np.newaxis]
                distance = np.linalg.norm(y - y[i, :], axis = 1)
            elif self.mode == "combi":
                if len(y.shape) == 1:
                    y = y[:, np.newaxis]
                distance = np.power(np.linalg.norm(X - X[i, :], axis = 1), 2) + np.power(np.linalg.norm(y - y[i, :], axis = 1), 2)
            neighbour_index[i, :] = np.argsort(distance)[:self.neighbours]

        probs = np.full(size, (self.clusters / size) * (self.batch_size / (self.clusters * self.neighbours)))
        for i in range(size):
            probs[i] *= np.sum([i in neighbour_index[j, :] for j in range(size)])

        if (self.batches is not None) and (type(self.batches) == type(1)):
            batch = np.zeros((self.batches, self.batch_size), dtype = "int")
            for i in tqdm(range(self.batches), leave = False, desc = "Determining batches"):
                cluster_index = np.random.choice(np.arange(size), size = self.clusters, replace = False)
                for j in range(self.clusters):
                    batch[i, j*(self.batch_size // self.clusters):(j+1)*(self.batch_size // self.clusters)] = np.random.choice(neighbour_index[cluster_index[j], :], size = self.batch_size // self.clusters, replace = False)
        else:
            valid = np.arange(0, size)
            batch = np.zeros((size // self.clusters, self.batch_size), dtype = "int")
            for i in tqdm(range(batch.shape[0]), leave = False, desc = "Determining batches"):
                cluster_index = np.random.choice(np.arange(size), size = self.clusters, replace = False)
                valid = np.delete(valid, np.isin(valid, cluster_index))
                for j in range(self.clusters):
                    batch[i, j*(self.batch_size // self.clusters):(j+1)*(self.batch_size // self.clusters)] = np.random.choice(neighbour_index[cluster_index[j], :], size = self.batch_size // self.clusters, replace = False)

        return batch, 1. / probs

class Act(torch.nn.Module):
    def __init__(self, threshold = -25):
        super().__init__()
        self.threshold = threshold

    def forward(self, l):
        if l.shape[1] > 1:
            return torch.cat([l[:, :-1], torch.unsqueeze(-torch.nn.Threshold(self.threshold, self.threshold)(-l[:, -1]), -1)], dim = 1)
        else:
            return -torch.nn.Threshold(self.threshold, self.threshold)(-l)

class PartialReLU(torch.nn.Module):
    def __init__(self, size, dims, bound = 0):
        super().__init__()
        self.dims = np.zeros(size, dtype = "bool")
        self.dims[dims] = True

        if isinstance(bound, list):
            if len(bound) < len(dims):
                for i in range(len(dims) - len(bound)):
                    bound.append(0)
            self.bound = bound[:len(dims)]
        else:
            self.bound = bound

    def forward(self, l):
        l[:, self.dims] = torch.clamp(l[:, self.dims], min = self.bound)
        return l

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

class MyScaler():
    def __init__(self, scaler) -> None:
        self.mean_ = scaler.mean_
        self.scale_ = scaler.scale_
        self.var_ = scaler.var_

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return self.scale_ * x + self.mean_

class NonScaler():

    def __init__(self, shift = 0):
        self.scale_  = 1
        self.shift_ = shift

    def fit(self, X):
        return self

    def transform(self, X):
        if torch.is_tensor(X):
            return X.numpy() + self.shift_
        else:
            return X + self.shift_

    def inverse_transform(self, X):
        if torch.is_tensor(X):
            return X.numpy() - self.shift_
        else:
            return X - self.shift_

class MAVScaler():
    def __init__(self, shift = False):
        self.mean = 0
        self.MAV = 0
        self.shift = shift

    def fit(self, data):
        self.mean = np.mean(data)
        self.MAV = np.mean(np.abs(data))
        return self

    def transform(self, data):
        return (np.array(data) - self.mean) / self.MAV if self.shift else np.array(data) / self.MAV

    def inverse_transform(self, data):
        return self.mean + np.array(data) * self.MAV if self.shift else np.array(data) * self.MAV