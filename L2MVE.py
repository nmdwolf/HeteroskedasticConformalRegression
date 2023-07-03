import scipy.stats as st

from MeanVarianceModel import MeanVarianceModel

import torch
import torch.nn as nn

from INIT import apply_dropout
from Loss import GaussianLoss

class L2MVE(MeanVarianceModel):

    def __init__(self, params, input = 1):
        super().__init__("MVE", params)

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

    def apply_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
        
    def eval(self):
        self.model.eval()
        self.model.apply(apply_dropout)

    def internal(self, batch_x, batch_y, weights, num = 50):
        preds = self.model(batch_x)
        # out = torch.zeros((batch_x.shape[0], 2))
        # for i in range(num):
        #     preds = self.model(batch_x)
        #     out[:, 0] += preds[:, 0]
        #     out[:, 1] += torch.exp(preds[:, 1]) + torch.square(preds[:, 0])
        # preds = out / num

        return GaussianLoss(preds[:, 0], batch_y, preds[:, 1], weights)
        # return GaussianLoss(preds[:, 0], batch_y, torch.clamp(preds[:, 1] - torch.square(preds[:, 0]), min = 0), weights)

    def mve_ensemble(self, X, num = 50):
        out = torch.zeros((X.shape[0], 2))
        for i in range(num):
            preds = self.model(X)
            out[:, 0] += preds[:, 0]
            out[:, 1] += torch.exp(preds[:, 1]) + torch.square(preds[:, 0])
        out = out / num

        out[:, 1] = torch.clamp(out[:, 1] - torch.square(out[:, 0]), min = 0) # zero-clipping to avoid NaN results in square root
        return out

    def predict(self, X, std = False, alpha = None, ensemble = 50):

        self.eval()
        with torch.no_grad():
            preds = self.mve_ensemble(X, ensemble)
            
            sig = torch.sqrt(preds[:, 1])
            if std:
                preds[:, 1] = sig

            if alpha is not None:
                
                z = st.norm.ppf(1 - (alpha / 2))
                lower = preds[:, 0] - z * sig
                upper = preds[:, 0] + z * sig
                
                return preds, torch.stack([lower, upper], dim = -1)
            
            else:
                return preds

    def predict_(self, X, ensemble = 50):
        
        out = torch.zeros((X.shape[0], 2))
        for i in range(ensemble):
            preds = self.model(X)
            out[:, 0] += preds[:, 0]
            out[:, 1] += torch.exp(preds[:, 1]) + torch.square(preds[:, 0])
        out = out / ensemble

        return torch.stack([out[:, 0], torch.clamp(out[:, 1] - torch.square(out[:, 0]), min = 0.)], dim = -1) # zero-clipping to avoid NaN results in square root