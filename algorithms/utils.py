import numpy as np
import math
from numpy import histogram as hist


def chain_error(ref_chain, evaluated_chain):
    """
    Given two one dimensional chains - this function evaluates the area difference in the histograms
    """
    ref_values, ref_bins = hist(ref_chain, bins="auto", normed=True)
    alg_values, _ = hist(evaluated_chain, bins=ref_bins, normed=True)
    alg_error = np.sum(np.diff(ref_bins) * abs(ref_values - alg_values))
    return alg_error


def normalLogLhd(x, mu, sigma):
    """
    return an array of Gaussian log likelihoods up to a constant
    """
    return -(x - mu) ** 2 / (2 * sigma ** 2) - np.log(sigma)


def l2_norm_squared(vec: np.array):
    return np.sum(vec * vec)
