import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kstest, spearmanr, pearsonr
import scipy.stats as st
from sklearn.metrics import r2_score

from CP import *

class Logger():
    
    def __init__(self, data, folder, datasource, seed, alpha = 0.1, var_bins = 3, verbose = False, synthetic = False):
        
        self.data = data
        self.folder = folder
        self.datasource = datasource
        self.alpha = alpha
        self.var_bins = var_bins
        self.synthetic = synthetic
        self.seed = seed

        self.models = []

        self.covs = {}
        self.widths = {}
        self.r2s = {}
        
        self.conditional_covs = {}
        self.conditional_r2s = {}
        self.conditional_widths = {}
        self.conditional_scores = {}
        
        self.score = None
        self.verbose = verbose

    def save(self, name, r2, covs, widths):

        self.r2s[name] = r2
        self.covs[name] = np.mean(covs.numpy())
        if torch.is_tensor(widths):
            self.widths[name] = np.mean(widths.numpy())
        else:
            self.widths[name] = widths
        
        if self.verbose:
            print("Item " + str(len(self.covs.keys())) + " (" + name + ") registered")
            
    def save_conditional(self, name, r2, covs, widths, scores = None):
        
        if name not in self.conditional_covs.keys():
            self.conditional_r2s[name] = []
            self.conditional_covs[name] = []
            self.conditional_widths[name] = []
            self.conditional_scores[name] = []
            
        self.conditional_covs[name].append(np.mean(covs.numpy()))
        self.conditional_r2s[name].append(r2)
        if torch.is_tensor(widths):
            self.conditional_widths[name].append(np.mean(widths.numpy()))
        else:
            self.conditional_widths[name].append(widths)

        if scores is not None:
            self.conditional_scores[name].append(scores.numpy())
                
        if self.verbose:
            print("Item " + str(len(self.conditional_covs.keys())) + "|" + str(len(self.conditional_covs[name])) + " (" + name + ") registered")

    def to_array(self):
        r2s = np.stack([self.r2s[name] for name in self.r2s.keys()], axis = -1)
        covs = np.stack([self.covs[name] for name in self.covs.keys()], axis = -1)
        widths = np.stack([self.widths[name] for name in self.covs.keys()], axis = -1)

        return r2s, covs, widths, list(self.covs.keys())
    
    def to_array_conditional(self):
        covs = np.zeros((len(self.conditional_covs.keys()), len(list(self.conditional_covs.items())[0][1])))
        r2s = np.zeros_like(covs)
        widths = np.zeros_like(covs)
        for i, key in enumerate(self.conditional_covs.keys()):
            covs[i, :] = self.conditional_covs[key]
            r2s[i, :] = self.conditional_r2s[key]
            widths[i, :] = self.conditional_widths[key]

        return r2s, covs, widths, list(self.conditional_covs.keys())

    def run_KS(self, compare_outer = True):
        names = []
        result = []
        for name in self.conditional_scores.keys():
            if len(self.conditional_scores[name]) > 0:
                names.append(name)
                if compare_outer:
                    tests = [kstest(self.conditional_scores[name][i], self.conditional_scores[name][i+1])[1] for i in range(len(self.conditional_scores[name]) - 1)]
                    tests.append(kstest(self.conditional_scores[name][0], self.conditional_scores[name][-1])[1])
                    result.append(np.array(tests))
                else:
                    result.append(np.array([kstest(self.conditional_scores[name][i], self.conditional_scores[name][i+1])[1] for i in range(len(self.conditional_scores[name]) - 1)]))

                plt.figure(figsize = (8, 8))
                for i in range(len(self.conditional_scores[name])):
                    scores = self.conditional_scores[name][i]
                    plt.plot(np.sort(scores))
                plt.savefig(self.folder + self.datasource + "_KS-" + name.replace("*", "&") + ".svg", dpi = 320, bbox_inches = "tight")
                plt.close()

        return np.stack(result, axis = 0), names

    def internal(self, model, name, flag = False):

        val_preds = model.predict(self.data["X_val"])
        var = np.sort(val_preds[:, 1])

        splits = [-1e10]
        for i in range(self.var_bins - 1):
            splits.append(var[(i + 1) * (len(var) // self.var_bins)])
        splits.append(1e10)
        
        if flag:
            score = SwissKnife(lambda x: model.predict(x, alpha = self.alpha))
            score_conditional = SwissKnife(lambda x: model.predict(x, alpha = self.alpha))
            score.predict(self.data["X_test"])
            preds, pi = score.preds, score.pi
        else:
            score = CachedSwissKnife(lambda x: model.predict(x, alpha = self.alpha), self.data["X_val"], self.data["y_val"])
            preds, pi = model.predict(self.data["X_test"], alpha = self.alpha)

        r2 = r2_score(self.data["y_test"], preds[:, 0])
        self.save(name + "-base", r2, (self.data["y_test"] >= pi[:, 0]) & (self.data["y_test"] <= pi[:, 1]), pi[:, 1] - pi[:, 0])

        test = [splits[i+1] == splits[i+2] for i in range(self.var_bins-1)]
        override = True in test

        for i in range(self.var_bins):
            val_index = (val_preds[:, 1] >= splits[i]) & (val_preds[:, 1] <= splits[i + 1])
            test_index = (preds[:, 1] >= splits[i]) & (preds[:, 1] <= splits[i + 1])

            if override:
                val_index = range(val_preds.shape[0])
                test_index = range(preds.shape[0])

            X_val = self.data["X_val"][val_index, :]
            y_val = self.data["y_val"][val_index]
            y_test = self.data["y_test"][test_index]

            r2 = r2_score(y_test, preds[test_index, 0])
            self.save_conditional(name + "-base", r2, (y_test >= pi[test_index, 0]) & (y_test <= pi[test_index, 1]), \
                pi[test_index, 1] - pi[test_index, 0])

            if flag:
                # score = MSEScore(lambda x: model.predict(x)[:, 0])
                # crit = score.quantile(self.data["X_val"], self.data["y_val"], alpha)
                crit = score.quantile(self.data["X_val"], self.data["y_val"], self.alpha, mode = "MSE")
                self.save(name + "-PointCP", r2, (self.data["y_test"] >= preds[:, 0] - crit) & (self.data["y_test"] <= preds[:, 0] + crit), \
                2 * crit)
                self.save_conditional(name + "-mPointCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit, score(self.data["X_val"], self.data["y_val"], mode = "MSE")[val_index])

                # score = IntervalScore(lambda x: model.predict(x, alpha = alpha)[1])
                # crit = score.quantile(self.data["X_val"], self.data["y_val"], alpha)
                crit = score.quantile(self.data["X_val"], self.data["y_val"], self.alpha, mode = "interval")
                self.save(name + "-IntCP", r2, (self.data["y_test"] >= pi[:, 0] - crit) & (self.data["y_test"] <= pi[:, 1] + crit), \
                pi[:, 1] - pi[:, 0] + 2 * crit)
                self.save_conditional(name + "-mIntCP", r2, (y_test >= pi[test_index, 0] - crit) & (y_test <= pi[test_index, 1] + crit), \
                    pi[test_index, 1] - pi[test_index, 0] + 2 * crit, score(self.data["X_val"], self.data["y_val"], mode = "interval")[val_index])

                # score = NormalizedMSEScore(lambda x: model.predict(x, std = True))
                # crit = score.quantile(self.data["X_val"], self.data["y_val"], alpha) * preds[test_index, 1]
                crit = score.quantile(self.data["X_val"], self.data["y_val"], self.alpha, mode = "nMSE") * np.sqrt(preds[:, 1])
                self.save(name + "-NCP", r2, (self.data["y_test"] >= preds[:, 0] - crit) & (self.data["y_test"] <= preds[:, 0] + crit), \
                2 * crit)
                self.save_conditional(name + "-mNCP", r2, (y_test >= preds[test_index, 0] - crit[test_index]) & (y_test <= preds[test_index, 0] + crit[test_index]), 2 * crit[test_index], score(self.data["X_val"], self.data["y_val"], mode = "nMSE")[val_index])

                # score = MSEScore(lambda x: model.predict(x)[:, 0])
                # crit = score.quantile(X_val, y_val, alpha)
                crit = score_conditional.quantile(X_val, y_val, self.alpha, mode = "MSE")
                self.save_conditional(name + "-PointCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit)

                # score = IntervalScore(lambda x: model.predict(x, alpha = alpha)[1])
                # crit = score.quantile(X_val, y_val, alpha)
                crit = score_conditional.quantile(X_val, y_val, self.alpha, mode = "interval")
                self.save_conditional(name + "-IntCP", r2, (y_test >= pi[test_index, 0] - crit) & (y_test <= pi[test_index, 1] + crit), \
                    pi[test_index, 1] - pi[test_index, 0] + 2 * crit)

                # score = NormalizedMSEScore(lambda x: model.predict(x, std = True))
                # crit = score.quantile(X_val, y_val, alpha) * preds[test_index, 1]
                crit = score_conditional.quantile(X_val, y_val, self.alpha, mode = "nMSE") * np.sqrt(preds[test_index, 1])
                self.save_conditional(name + "-NCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit)
            else:
                crit = score.quantile(range(val_preds.shape[0]), self.alpha, mode = "MSE")
                self.save(name + "-PointCP", r2, (self.data["y_test"] >= preds[:, 0] - crit) & (self.data["y_test"] <= preds[:, 0] + crit), \
                2 * crit)
                self.save_conditional(name + "-mPointCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit, score(val_index, mode = "MSE"))

                crit = score.quantile(range(val_preds.shape[0]), self.alpha, mode = "interval")
                self.save(name + "-IntCP", r2, (self.data["y_test"] >= pi[:, 0] - crit) & (self.data["y_test"] <= pi[:, 1] + crit), \
                pi[:, 1] - pi[:, 0] + 2 * crit)
                self.save_conditional(name + "-mIntCP", r2, (y_test >= pi[test_index, 0] - crit) & (y_test <= pi[test_index, 1] + crit), \
                    pi[test_index, 1] - pi[test_index, 0] + 2 * crit, score(val_index, mode = "interval"))

                crit = score.quantile(range(val_preds.shape[0]), self.alpha, mode = "nMSE") * np.sqrt(preds[:, 1])
                self.save(name + "-NCP", r2, (self.data["y_test"] >= preds[:, 0] - crit) & (self.data["y_test"] <= preds[:, 0] + crit), \
                2 * crit)
                self.save_conditional(name + "-mNCP", r2, (y_test >= preds[test_index, 0] - crit[test_index]) & (y_test <= preds[test_index, 0] + crit[test_index]), 2 * crit[test_index], score(val_index, mode = "nMSE"))

                crit = score.quantile(val_index, self.alpha, mode = "MSE")
                self.save_conditional(name + "-PointCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit)

                crit = score.quantile(val_index, self.alpha, mode = "interval")
                self.save_conditional(name + "-IntCP", r2, (y_test >= pi[test_index, 0] - crit) & (y_test <= pi[test_index, 1] + crit), \
                    pi[test_index, 1] - pi[test_index, 0] + 2 * crit)

                crit = score.quantile(val_index, self.alpha, mode = "nMSE") * np.sqrt(preds[test_index, 1])
                self.save_conditional(name + "-NCP", r2, (y_test >= preds[test_index, 0] - crit) & (y_test <= preds[test_index, 0] + crit), \
                    2 * crit)

        return preds[:, 1], pi[:, 1] - pi[:, 0]

    def register(self, model, name = None):

        if name is None:
            name = model.name

        if name not in [m[0] for m in self.models]:
            self.models.append((name, model))

    def run(self):
        variances = []
        pis = []
        with torch.no_grad():
            for name, model in self.models:
                var, pi = self.internal(model, name)
                variances.append(var)
                pis.append(pi)

        if self.synthetic:
            variance_plot(self.data["var_test"], variances, pis, self.data["scaler_y"], [m[0] for m in self.models], save = True, title = str(self.data["seed"]), folder = self.folder, datasource = self.datasource)

def variance_plot(true, pred, pi, scaler, labels, folder, datasource, alpha = 0.1, save = False, title = None) -> None:
    
    high = torch.max(true)

    file = open(folder + "VARIANCE/" + datasource + "-rank.txt", "w")
    
    # plt.figure(figsize = (14, 14))
    plt.figure()
    plt.title("Variance plot")
    
    for i, label in enumerate(labels): 
        if i >= len(pred):
            break
        high = max(high, torch.max(pred[i] * (scaler.scale_ ** 2)))
        plt.scatter(true, pred[i] * (scaler.scale_ ** 2), label = label, s = 1)
        file.write(label + "|" + str(pearsonr(true, pred[i] * (scaler.scale_ ** 2)).statistic) + "|" + str(spearmanr(true, pred[i] * (scaler.scale_ ** 2)).correlation) + "\n")
        
    # plt.plot([0, high], [0, high], linestyle = "--", color = "red")
    if "Oracle" not in labels:
        plt.scatter(true, true, label = "True", s = 2)

    plt.legend()
    
    if save:
        plt.savefig(folder + "VARIANCE/" + datasource + "_variance" + title + ".svg", dpi = 320, bbox_inches = "tight")
        plt.savefig(folder + "VARIANCE/" + datasource + "_variance" + title + ".png", dpi = 500, bbox_inches = "tight")
    else:
        plt.show()
        
    plt.close()

    plt.figure(figsize = (14, 14))
    plt.title("Widths plot")
    
    for i, label in enumerate(labels): 
        if i >= len(pred):
            break
        plt.scatter(2 * st.norm.ppf(1 - (alpha / 2)) * np.sqrt(true), pi[i] * scaler.scale_, label = label)
    
    plt.scatter(2 * st.norm.ppf(1 - (alpha / 2)) * np.sqrt(true), 2 * st.norm.ppf(1 - (alpha / 2)) * np.sqrt(true), label = "True")
    plt.legend()
    
    if save:
        plt.savefig(folder + "VARIANCE/" + datasource + "_width" + title + ".svg", dpi = 320, bbox_inches = "tight")
        plt.savefig(folder + "VARIANCE/" + datasource + "_width" + title + ".png", dpi = 500, bbox_inches = "tight")
    else:
        plt.show()
        
    plt.close()