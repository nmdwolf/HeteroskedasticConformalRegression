import scipy.stats as st

import torch
import torch.nn as nn

from INIT import *
from UQModel import UQModel
from Loss import PinballLoss

class QR(UQModel):

    def __init__(self, alpha, params, input = 1):
        super().__init__("QR", params)

        self.alpha = alpha
        self.quantiles = [self.alpha / 2, 1 - self.alpha / 2, 0.5]

        self.model = nn.Sequential(
            nn.Linear(input, self.dim),
            *([nn.ReLU() if self.act == "ReLU" else nn.ELU(), nn.Dropout(self.drop), nn.Linear(self.dim, self.dim)] * self.hidden_layers),
            nn.ReLU() if self.act == "ReLU" else nn.ELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.dim, len(self.quantiles))
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def eval(self):
        self.model.eval()

    def internal(self, batch_x, batch_y, weights):

        pi = self.model(batch_x)
        return PinballLoss(pi, batch_y, self.quantiles)

    def predict(self, X, std = False, alpha = None):

        self.model.eval()
        with torch.no_grad():

            pi = self.model(X)
            
            median = pi[:, 2]
            
            z = st.norm.ppf(1 - self.alpha / 2)
            sig = (pi[:, 1] - pi[:, 0]) / (2 * z)

            if not std:
                sig = sig ** 2
            
            if not alpha:
                return torch.stack([median, sig], dim = -1)
            else:
                return torch.stack([median, sig], dim = -1), pi