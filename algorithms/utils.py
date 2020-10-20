import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import histogram as hist


def chain_error(ref_chain, evaluated_chain, show_hists=False, title=None):
    """
    Given two one dimensional chains - this function evaluates the area difference in the histograms

    Splits the area into optional param `bins` number of bins
    """
    plt.clf()
    ref_values, ref_bins, _ = plt.hist(
        ref_chain, bins="auto", density=True, alpha=0.5, label="Ref"
    )
    alg_values, _, _ = plt.hist(
        evaluated_chain, bins=ref_bins, density=True, alpha=0.5, label="Alg"
    )
    # COmputing difference in Reimann sums of histograms
    alg_error = np.sum(
        np.diff(ref_bins) * abs(np.array(ref_values) - np.array(alg_values))
    )
    if show_hists:
        plt.legend(loc="upper right")
        plt.suptitle(title)
        plt.show()
    return alg_error


def normalLogLhd(x, mu, sigma):
    """
    return an array of Gaussian log likelihoods up to a constant
    """
    return -(x - mu) ** 2 / (2 * sigma ** 2) - np.log(sigma)


def l2_norm_squared(vec: np.array):
    return np.sum(vec * vec)
