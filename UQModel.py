from abc import ABC, abstractmethod

from INIT import *

import torch

from itertools import product

class UQModel():

    def __init__(self, name, params = {}):
        self.name = name

        self.decay = DECAY if not "decay" in params.keys() else params["decay"]
        self.dim = DIM if not "hidden_dim" in params.keys() else params["hidden_dim"]
        self.drop = DROP if not "drop" in params.keys() else params["drop"]
        self.rate = RATE if not "rate" in params.keys() else params["rate"]
        self.hidden_layers = LAYERS if not "hidden_layers" in params.keys() else params["hidden_layers"]
        self.act = "ELU" if not "act" in params.keys() else params["act"]
        self.verbose = True if not "verbose" in params.keys() else params["verbose"]
        self.optimizer = torch.optim.Adam
        self.options = params

    def __getitem__(self, key):
        if key == "decay":
            return self.decay
        elif key == "hidden_dim" or key == "dim":
            return self.dim
        elif key == "drop":
            return self.drop
        elif key == "epochs":
            return self.epochs
        elif key == "rate":
            return self.rate
        elif key in self.options.keys():
            return self.options[key]
        else:
            return None

    def __setitem__(self, key, value):
        self.options[key] = value

    @abstractmethod
    def internal(self, batch_X, batch_y, weights):
        pass

    def train(self, X, y, batch_maker, epochs = EPOCHS, target = "cpu"):
        self.model.to(device = target)

        X = X.requires_grad_(False).to(device = target)
        y = y.requires_grad_(False).to(device = target)
        batch, weights = batch_maker.sample(X, y)
        
        self.optimizer = self.optimizer(self.model.parameters(), lr = self.rate, weight_decay = self.decay)

        self.model.train()
        for e, b in product(range(epochs), range(batch.shape[0])):

            self.optimizer.zero_grad()

            index = np.unique(batch[b, :]) # np.unique is to make this work with HT estimation
            batch_x = X[index, :]
            batch_y = y[index]

            loss = self.internal(batch_x, batch_y, weights[index])

            loss.backward()
            self.optimizer.step()

        self.model.to(device = "cpu")

    @abstractmethod
    def predict(self, X, std = False, alpha = None):
        pass

    @abstractmethod
    def eval(self):
        pass
