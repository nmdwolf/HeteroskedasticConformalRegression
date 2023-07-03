import math, copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
from INIT import conditional_UR_1d, MyScaler
from UQModel import UQModel

class ConditionalDistribution():

    def __init__(self, to_torch = True, output_stats = True):
        self.torch = to_torch
        self.output = output_stats

    def sample(self, X, seed = None):
        if seed is not None:
            np.random.seed(seed)

    def description(self):
        pass

class MeanVarianceDistribution(ConditionalDistribution):

    def __init__(self, mean, variance, to_torch = False, output_stats = True):
        super().__init__(to_torch, output_stats)
        self.mean = mean
        self.variance = variance

        self.fmean = lambda x: torch.from_numpy(self.mean(x.numpy()))
        self.fvar = lambda x: torch.from_numpy(self.variance(x.numpy()))

    def sample(self, X, seed = None):
        super().sample(X, seed)

        if torch.is_tensor(X):
            X = X.numpy()

        mean = self.mean(X)
        variance = self.variance(X)

        if not self.output:
            return torch.Tensor(self.internal(mean, variance)) if self.torch else self.internal(mean, variance)
        else:
            return torch.Tensor(self.internal(mean, variance)) if self.torch else self.internal(mean, variance), mean, variance

    def internal(self, mean, variance):
        pass

    def getFunctions(self):
        return self.fmean, self.fvar

class ConditionalNormal(MeanVarianceDistribution):

    def internal(self, mean, variance):
        return mean + np.random.normal(loc = 0, scale = np.sqrt(variance))

    def description(self):
        return "normal"

class ConditionalUniform(MeanVarianceDistribution):

    def internal(self, mean, variance):
        dev = np.sqrt(3 * variance)
        return mean + np.random.uniform(-dev, dev)

    def description(self):
        return "uniform"

class ConditionalLaplace(MeanVarianceDistribution):

    def internal(self, mean, variance):
        return mean + np.random.laplace(loc = 0, scale = np.sqrt(variance / 2))

    def description(self):
        return "Laplace"

class ConditionalExponential(MeanVarianceDistribution):

    def internal(self, mean, variance):
        temp = copy.deepcopy(mean)
        mean += np.sqrt(variance)
        return temp + np.random.exponential(scale = np.sqrt(variance))

    def description(self):
        return "exp"

# class ConditionalVonMises(MeanVarianceDistribution):

#     def internal(self, mean, variance):
#         return np.random.vonmises(mu = mean, kappa = np.sqrt(variance)) #IMPLEMENTATION TO BE CORRECTED

#     def description(self):
#         return "mises"

class ConditionalSawtooth(MeanVarianceDistribution):

    def internal(self, mean, variance):
        low = mean - 2 * np.sqrt(2 * variance)
        high = mean + np.sqrt(2 * variance)

        sample = np.random.triangular(low, high, 2 * high - low)
        sample[sample > high] -= 2 * (sample[sample > high] - high[sample > high])

        return sample

    def description(self):
        return "sawtooth"

class ConditionalBeta(MeanVarianceDistribution): # CHECK!!!

    def __init__(self, mean, variance, to_torch=False, output_stats=True):
        super().__init__(mean, variance, to_torch, output_stats)
        
        self.fmean = lambda x: torch.from_numpy(self.mean(x.numpy()) + np.sqrt(self.variance(x.numpy())) * np.full((x.shape[0],), 0.5))
        self.fvar = lambda x: torch.from_numpy(self.variance(x.numpy())) / 8

    def internal(self, mean, variance):
        m = np.full_like(mean, 0.5)
        v = np.full_like(variance, 1 / 8)
        blob = (m * (1 - m)) / v - 1
        noise = np.random.beta(m * blob, (1 - m) * blob)

        temp = copy.deepcopy(mean)
        temp2 = copy.deepcopy(variance)
        mean += np.sqrt(variance) * m
        variance /= 8

        return temp + temp2 * noise

    def description(self):
        return "beta"

class ConditionalTetris(MeanVarianceDistribution):

    def __init__(self, mean, variance, to_torch=False, output_stats=True):
        super().__init__(mean, variance, to_torch, output_stats)

        self.fvar = lambda x: torch.from_numpy(self.variance(x.numpy())) + (5 / 48)

    def internal(self, mean, variance):
        variance += 5 / 48
        a = mean - 0.2 * np.sqrt(15 * variance - 1) + 0.4
        b = 0.2 * (4 * np.sqrt(15 * variance - 1) - 3)
        mode = np.random.choice([0, 1], len(mean))

        return mode * np.random.uniform(a - 1, a) + (1 - mode) * np.random.uniform(a, a + b)

    def description(self):
        return "tetris"

class ConditionalBimodal(MeanVarianceDistribution):

    def __init__(self, mean, variance, spread = 2, to_torch=False, output_stats=True):
        super().__init__(mean, variance, to_torch, output_stats)

        self.spread = spread
        self.fvar = lambda x: torch.from_numpy(self.variance(x.numpy())) + (self.spread / 2) ** 2

    def internal(self, mean, variance):
        choice = (mean <= 2).astype(int)
        choice = ((-1) ** choice) * self.spread / 2
        var = copy.deepcopy(variance)
        variance += (self.spread / 2) ** 2

        return mean + np.random.normal(loc = choice, scale = np.sqrt(var))

    def description(self):
        return "bimodal"

# class ConditionalScalingPyramid(MeanVarianceDistribution):

#     def internal(self, mean, variance):

#         def f(y, X):
#             a = self.mean_func(X)
#             b = 0.5 * (np.sqrt(12 * (self.variance(X) + 1) - 3) + 1)
#             H = 0.5 / (b + 1)
#             return H * (((y >= (a - b)) & (y < (a - 1))) + 2 * ((y >= (a - 1)) & (y < (a + 1))) + ((y >= (a + 1)) & (y < (a + b))))

#         a = mean
#         b = 0.5 * (np.sqrt(12 * (variance + 1) - 3) + 1)
#         y = conditional_UR_1d(f, X, min = int(np.min(a - 2 * b)), max = int(np.max(a + 2 * b)))
#         return y, mean, variance + 1

#     def description(self):
#         return "scaling_pyramid"

# def construct(X, y):
#             var = lmbda * var_func(X) + 1

#             def f(y, X):
#                 a = mean_func(X)
#                 b = 0.5 * (np.sqrt(12 * (lmbda * var_func(X) + 1) - 3) + 1)
#                 H = 0.5 / (b + 1)
#                 return H * (((y >= (a - b)) & (y < (a - 1))) + 2 * ((y >= (a - 1)) & (y < (a + 1))) + ((y >= (a + 1)) & (y < (a + b))))

#             a = mean_func(X)
#             b = 0.5 * (np.sqrt(12 * (lmbda * var_func(X) + 1) - 3) + 1)
#             y = conditional_UR_1d(f, X, min = int(np.min(a - 2 * b)), max = int(np.max(a + 2 * b)))
#             return y, var

def getStatistics(choice: str, lmbda: float, mean_lmbda: float = 1., params: dict = None, lower: float = 0.1):

    mean_func = lambda x: mean_lmbda * np.mean(x, axis = 1)

    if choice[:3] == "dim":

        # var_func = lambda x: np.abs(np.mean(x, axis = 1))
        # var_func = lambda x: np.abs(np.sum(np.square(x - 1.), axis = 1))

        var_func = lambda x: lmbda * np.abs(x[:, int(choice[3:])] + lower)

    elif choice == "diag":

        var_func = lambda x: lmbda * np.abs(np.mean(x[:, 2], axis = 1) + lower)

    elif choice == "antidiag":

        var_func = lambda x: lmbda * np.abs(x[:, 0] - x[:, 1] + lower)
        
    elif choice == "parametric":

        #mean_func = lambda x: mean_lmbda * params["mean"](x)
        var_func = lambda x: lmbda * np.sqrt(np.abs(np.mean(x, axis = 1)))

    elif choice == "constant":

        var_func = lambda x: lmbda * np.ones(x.shape[0])
        
    elif choice[:2] == "cm": # 'cm': constant mean
        
        mean_func = lambda x: np.full(x.shape[0], mean_lmbda)
        var_func = lambda x: lmbda * np.abs(np.mean(x, axis = 1))

    elif choice == "mean":

        # var_func = lambda x: lmbda * np.mean(np.abs(x) + lower, axis = 1)
        var_func = lambda x: lmbda * np.mean(np.abs(x), axis = 1)

    elif choice == "square":

        var_func = lambda x: lmbda * (np.square(np.mean(np.abs(x), axis = 1)) + lower)

    return mean_func, var_func

def init(feature_choice: str, conditional_sampler: ConditionalDistribution, seed: int = 2, verbose: bool = False, \
    scaler = StandardScaler(), scaler_y = StandardScaler(), cp_mode: bool = True, overlap: int = 0, params = {}, to_torch: bool = True):

    X, y = None, None
    X_train, X_val, X_test = None, None, None
    y_train, y_val, y_test = None, None, None
    train_var, val_var, test_var = None, None, None
    additional = None

    if "size" in params.keys():
        train_size = 3 * params["size"]
        size = params["size"]
    else:
        train_size = 1500
        size = 500

    if "dim" in params.keys():
        dim = params["dim"]
    else:
        dim = 2

    if "var_dim" not in params.keys():
        params["var_dim"] = 0

    if "lambda" in params.keys():
        lmbda = params["lambda"]
    else:
        lmbda = 0.1

    if "high" in params.keys():
        high = params["high"]
    else:
        high = 10
        
    if "low" in params.keys():
        low = params["low"]
    else:
        low = 0
        
    np.random.seed(seed)
    
    if feature_choice == "uniform" or feature_choice == None:
    
        X_train = np.random.uniform(low = 0, high = high, size = (train_size, dim))
        X_val = np.random.uniform(low = 0, high = high, size = (size, dim))
        X_test = np.random.uniform(low = 0, high = high, size = (size, dim))
        
    elif feature_choice == "bimodal":
        
        split = high / 2
        
        modes = np.random.choice([0, 1], (train_size, dim))
        X_train = np.random.uniform(modes * split, (modes + 1) * split)
        
        modes = np.random.choice([0, 1], (size, dim))
        X_val = np.random.uniform(modes * split, (modes + 1) * split)
        
        modes = np.random.choice([0, 1], (size, dim))
        X_test = np.random.uniform(modes * split, (modes + 1) * split)

    elif feature_choice == "normal":

        # mean = (high - low,) * dim
        mean = (0,) * dim
        cov = np.diag((high - low,) * dim) / max(1, high // 10)

        X_train = np.random.multivariate_normal(mean, cov, size = train_size)
        X_val = np.random.multivariate_normal(mean, cov, size = size)
        X_test = np.random.multivariate_normal(mean, cov, size = size)
    
    additional = {"seed": seed}
    y_train, train_mean, train_var = conditional_sampler.sample(X_train)
    y_val, val_mean, val_var = conditional_sampler.sample(X_val)
    y_test, test_mean, test_var = conditional_sampler.sample(X_test)

    # if "bimodal_hetero" in choice:
    #     modes = np.random.choice([0, 1], train_size)
    #     y_train = modes * np.random.normal(loc = -mean_func(X_train), scale = np.sqrt(train_var))
    #     y_train += (1 - modes) * (np.random.exponential(scale = np.sqrt(train_var)) + mean_func(X_train))

    #     modes = np.random.choice([0, 1], size)
    #     y_val = modes * np.random.normal(loc = -mean_func(X_val), scale = np.sqrt(val_var))
    #     y_val += (1 - modes) * (np.random.exponential(scale = np.sqrt(val_var)) + mean_func(X_val))

    #     modes = np.random.choice([0, 1], size)
    #     y_test = modes * np.random.normal(loc = -mean_func(X_test), scale = np.sqrt(test_var))
    #     y_test += (1 - modes) * (np.random.exponential(scale = np.sqrt(test_var)) + mean_func(X_test))
    #     additional.update({"identifier": additional["identifier"] + "_bimodal_exp"})

    # elif "bimodal" in choice:
    #     shift = 2

    #     modes = np.random.choice([0, 1], train_size)
    #     # y_train = np.random.normal(loc = mean_func(X_train) * (-1) ** modes, scale = np.sqrt(train_var))
    #     y_train = np.random.normal(loc = mean_func(X_train) + (-1) ** modes * shift, scale = np.sqrt(train_var))

    #     modes = np.random.choice([0, 1], size)
    #     # y_val = np.random.normal(loc = mean_func(X_val) * (-1) ** modes, scale = np.sqrt(val_var))
    #     y_val = np.random.normal(loc = mean_func(X_val) + (-1) ** modes * shift, scale = np.sqrt(val_var))

    #     modes = np.random.choice([0, 1], size)
    #     # y_test = np.random.normal(loc = mean_func(X_test) * (-1) ** modes, scale = np.sqrt(test_var))
    #     y_test = np.random.normal(loc = mean_func(X_test) + (-1) ** modes * shift, scale = np.sqrt(test_var))
    #     additional.update({"identifier": additional["identifier"] + "_bimodal"})

    # elif "sine" in choice:
    #     func = lambda y, x: (y > -math.pi) * (y < math.pi) * (y + mean_func(x)) * np.sin(y) / (2 * math.pi)
    #     y_train = conditional_UR_1d(func, X_train)
    #     y_val = conditional_UR_1d(func, X_val)
    #     y_test = conditional_UR_1d(func, X_test)

    # elif "beta_noise" in choice:

    #     def construct(X, y):
    #         mean = 0.5
    #         var = 0.25 * np.exp(-0.01 * var_func(X))
    #         blob = (mean * (1 - mean)) / var - 1
    #         noise = np.random.beta(mean * blob, (1 - mean) * blob)
    #         return y + noise - mean, var

    #     y_train, train_var = construct(X_train, train_mean)
    #     y_val, val_var = construct(X_val, val_mean)
    #     y_test, test_var = construct(X_test, test_mean)
        
    # elif "mix" in choice:

    #     def construct(X, y):
    #         modes = np.random.choice([0, 1], X.shape[0])
    #         var = lmbda * var_func(X)
    #         var = modes * var + (1 - modes) * 3 * var
    #         noise = np.random.normal(mean_func(X), np.sqrt(var))
    #         noise = (-1) ** modes * np.abs(noise)
    #         return y + noise, 2 * var

    #     y_train, train_var = construct(X_train, train_mean)
    #     y_val, val_var = construct(X_val, val_mean)
    #     y_test, test_var = construct(X_test, test_mean)

    # elif "pyramid" in choice:

    #     def construct(X, y):
    #         var = lmbda * var_func(X) + 1
            
    #         def f(y, X):
    #             a = mean_func(X)
    #             b = 0.5 * (np.sqrt(15 * (lmbda * var_func(X) + 1) - 4) - 1)
    #             H = 0.4 / (b - 1)
    #             return H * ((y >= (a - b)) & (y < (a - 1))) + 0.1 * ((y >= (a - 1)) & (y < (a + 1))) + H * ((y >= (a + 1)) & (y < (a + b)))

    #         y = conditional_UR_1d(f, X)
    #         return y, var

    #     y_train, train_var = construct(X_train, train_mean)
    #     y_val, val_var = construct(X_val, val_mean)
    #     y_test, test_var = construct(X_test, test_mean)
    
    # plt.figure()
    # plt.hist(train_var, bins = 30)
    # plt.hist(val_var, bins = 30)
    # plt.hist(test_var, bins = 30)
    # plt.show()
    # plt.close()

    X = np.concatenate([X_train, X_val, X_test], axis = 0)
    y = np.concatenate([y_train, y_val, y_test])

    additional.update({"train_mean": train_mean, "val_mean": val_mean, "test_mean": test_mean})
    additional.update({"train_var": train_var, "val_var": val_var, "test_var": test_var})
            
    if not cp_mode:
        X_train = np.concatenate([X_train, X_val], axis = 0)
        y_train = np.concatenate([y_train, y_val], axis = 0)
        X_val = X_train
        y_val = y_train

    if to_torch:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

        additional["train_mean"] = torch.from_numpy(additional["train_mean"])
        additional["val_mean"] = torch.from_numpy(additional["val_mean"])
        additional["test_mean"] = torch.from_numpy(additional["test_mean"])
        additional["train_var"] = torch.from_numpy(additional["train_var"])
        additional["val_var"] = torch.from_numpy(additional["val_var"])
        additional["test_var"] = torch.from_numpy(additional["test_var"])

    additional["mean"] = torch.from_numpy(np.concatenate([additional["train_mean"], additional["val_mean"], additional["test_mean"]]))
    additional["var"] = torch.from_numpy(np.concatenate([additional["train_var"], additional["val_var"], additional["test_var"]]))
        
    settings = {"X": X, "y": y, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test, "features": X_train.shape[1], "feature_choice": feature_choice}
    settings.update(additional)
        
    return settings