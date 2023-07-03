import numpy as np
import scipy.stats as st

from UQModel import UQModel

import torch
import torch.nn as nn

from Loss import GaussianLoss

class MV(UQModel):

    def __init__(self, params, input = 1):
        super().__init__("MV", params)

        self.model = nn.Sequential(
            nn.Linear(input, self.dim),
            *([nn.ReLU() if self.act == "ReLU" else nn.ELU(), nn.Dropout(self.drop), nn.Linear(self.dim, self.dim)] * self.hidden_layers),
            nn.ReLU() if self.act == "ReLU" else nn.ELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.dim, 2)
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def internal(self, batch_x, batch_y, weights):

        preds = self.model(batch_x)

        return GaussianLoss(preds[:, 0], batch_y, preds[:, 1], weights)


    def predict(self, X, alpha = None, std = False):

        self.eval()
        with torch.no_grad():
            preds = self.model(X)
            preds[:, 1] = torch.exp(preds[:, 1])

            sig = np.sqrt(preds[:, 1])

            if std:
                preds[:, 1] = sig
            
            if alpha is not None:
                z = st.norm.ppf((1 - alpha) + (alpha / 2))
                lower = preds[:, 0] - z * sig
                upper = preds[:, 0] + z * sig
                return preds, torch.stack([lower, upper], dim = -1)
            else:
                return preds

    def eval(self):
        self.model.eval()