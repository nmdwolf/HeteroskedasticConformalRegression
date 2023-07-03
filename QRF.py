import scipy.stats as st

from UQModel import UQModel
from pyquantrf import QuantileRandomForestRegressor as RF
from INIT import *

class QRF(UQModel):

    def __init__(self, alpha, params, median = True):
        super().__init__("QRF", params)
        self.alpha = alpha
        self.median = median
        self.quantiles = [alpha / 2, 1 - (alpha / 2)] if not median else [alpha / 2, 1 - (alpha / 2), .5]

    def train(self, X, y, batch_maker, estimators = 100, min_samples_leaf = 10, threads = 1, target = "cpu"):
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(y):
            y = y.numpy()
            
        self.model = RF(n_estimators = estimators, nthreads = threads, min_samples_leaf = min_samples_leaf)
        self.model.fit(X, y)

    def predict(self, X, alpha = None, std = False):

        with torch.no_grad():
            if torch.is_tensor(X):
                X = X.numpy()

            preds = torch.from_numpy(self.model.predict(X, self.quantiles))
            
            if self.median:
                mean = preds[:, 2]
            else:
                mean = (preds[:, 0] + preds[:, 1]) / 2
            
            z = st.norm.ppf(1 - self.alpha / 2)
            sig = (preds[:, 1] - preds[:, 0]) / (2 * z)
            
            if not std:
                sig = sig ** 2
            
            if not alpha:
                return torch.stack([mean, sig], dim = -1)
            else:
                return torch.stack([mean, sig], dim = -1), preds