# Utility file for loss functions

import torch
import numpy as np

def CRPSLoss(ensemble, true, weights = None, regularization = 0):

    if weights is None:
        weights = np.ones(true.shape[0])

    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights).requires_grad_(False)

    mse = torch.mean(torch.abs(ensemble.T - true), dim = 0)
    quadratic = torch.zeros_like(true)
    for i in range(ensemble.shape[1]):
        for j in range(ensemble.shape[1]):
            if i != j:
                quadratic += torch.abs(ensemble[:, i] - ensemble[:, j])

    loss = torch.dot(weights, mse - quadratic / (2 * ensemble.shape[1] * ensemble.shape[1]))

    return loss

def GaussianLoss(mean, true, logvar, weights = None, regularization = 1e-3):

    if weights is None:
        weights = np.ones(true.shape[0])

    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights).float().requires_grad_(False)

    if not torch.is_tensor(logvar):
        logvar = torch.from_numpy(logvar).float().requires_grad_(False)

    mse = torch.square(true - mean)
    loss = torch.dot(weights, mse / (torch.exp(logvar) + regularization) + logvar) / torch.sum(weights)

    return loss

def MSELoss(preds, true, logvar, weights = None):

    if weights is None:
        weights = np.ones(true.shape[0])

    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights).requires_grad_(False)

    if not torch.is_tensor(logvar):
        logvar = torch.from_numpy(logvar).requires_grad_(False)

    mse = torch.square(true - preds)
    loss = torch.dot(weights, mse) / torch.sum(weights)

    return loss

def QuantileLoss(preds, true, q):

    diff = true - preds
    return torch.mean(torch.max((q - 1) * diff, q * diff))

def PinballLoss(preds, true, quantiles = [0.1, 0.9]):

    # Third index for median !!!
    return torch.mean(torch.stack([QuantileLoss(preds[:, i], true, quantiles[i]) for i in range(len(quantiles))]))