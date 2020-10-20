import math
import numpy as np
import numpy.random as npr
import random as ran
import time

from .settings import DEFAULT_MAX_CHAIN_LENGTH
from .settings import DEFAULT_TIMEOUT
from .utils import normalLogLhd
from .utils import l2_norm_squared


def normal_sufficient_stat(samples):
    """
    Compute a sufficient stat for our likelihood.

    In this case a likelihood is a normal dist with unknown mean and variance,
    thus sufficient stat is a tuple of sums and sums of squares.

    See proof: https://encyclopediaofmath.org/wiki/Sufficient_statistic
    """
    # clearly wrong cause we dont know std but w/e
    return np.array([np.sum(samples), np.sum(samples * samples)])


def symmetric_sample_proposal(sample_indices, N, k):
    """
    Perturbs indices that corresponds to moving from some random set of included samples of size n in from an overall 
    sample of size N by swapping uniformly random k variables from being included/excluded.
    """
    out_sample = set(range(N)) - set(sample_indices)
    in_samples_leaving = ran.sample(sample_indices, k)
    out_samples_joining = ran.sample(out_sample, k)
    in_sample = set(sample_indices) - set(in_samples_leaving) | set(out_samples_joining)
    return list(in_sample)


def iss_mcmc(
    inital_theta,
    x,
    n=None,
    k=None,
    eps=20,
    stepsize=0.5,
    time_budget=DEFAULT_TIMEOUT,
    chain_length=DEFAULT_MAX_CHAIN_LENGTH,
    getLogLhd=normalLogLhd,
    sufficient_stat=normal_sufficient_stat,
):
    """
    Perform an informed sub-sampling MCMC of size :param n: on :param samples:

    We discard the initial value in every chain.

    :param initial_theta: The initial array representing the start of the Markov Chain for theta     
    :param x: The data on which you are trying to fit a posterior.
    :param n: Optional. The size of the subsample - must be less than the length of :param x:.
            Default to sqrt of data size.
    :param k: Optional. How many samples to perturb when swapping indices in the sub-samples space proposal transition. 
            Default 10.
    :param eps: Optional. The Epsilon value to use in the odds function of ISS. 
            Default 20.
    :param time_budget: Optional. How long we're allowing the chain to run for in seconds. 
            Default 10mins.
    :param chain_length: Optional. Maximum chain length - used to reserve memory.
            Defaults 10k.
    :param getLogLhd: Optional. The function for computing log-likelihood of an iid observation.
            Default normal log likelihood.
    """
    # Helper functions

    # Initialisation
    N = len(x)
    theta = inital_theta
    statistic_on_full_sample = sufficient_stat(x)
    if n is None:
        n = int(math.sqrt(len(x)))
    if k is None:
        k = int(math.sqrt(n))
    if n > N or k > n or k > N - n:
        raise ValueError(f"N {N} n {n} k {k} causes mismatch in subsamples")
    scalar = len(x) / n
    print(f"Running ISS with n = {n}, k = {k}")

    def delta(subsample):
        subsample_stat = scalar * sufficient_stat(subsample)
        return (statistic_on_full_sample - subsample_stat, subsample_stat)

    # Need indices specifically for exploring the subsamples
    subsample_indices = ran.sample(range(N), n)

    stepsize = stepsize / np.sqrt(N)
    theta_chain = np.zeros((chain_length, 2))
    # We dont need to keep the sample chain - might be nice to keep a statistics one?
    # Suff stat here is 2D
    sample_stat_chain = np.zeros((chain_length, 2))
    theta_acceptance = 0.0
    sample_acceptance = 0.0
    accepted_theta = 0
    accepted_samples = 0

    # Delta should really be its own fn

    start_time = time.time()
    for i in range(chain_length):
        if time.time() - start_time > time_budget:
            print(
                f"Time budget consumed at chain step {i+1}, returning truncated result"
            )
            theta_chain = theta_chain[:i, :]
            break
        subsample = x[subsample_indices]

        # Worked weirdly well with this being n?
        prop_sample_indices = symmetric_sample_proposal(subsample_indices, N, k)
        prop_subsample = x[prop_sample_indices]

        current_sample_delta, stat_on_subsample = delta(subsample)
        proposal_sample_delta, prop_stat_on_subsample = delta(prop_subsample)

        log_odds = eps * (
            l2_norm_squared(current_sample_delta)
            - l2_norm_squared(proposal_sample_delta)
        )
        log_u = np.log(npr.rand())
        if log_u < log_odds:
            sample_stat_chain[i, :] = prop_stat_on_subsample
            subsample_indices = prop_sample_indices
            subsample = prop_subsample
            accepted_samples += 1
        else:
            sample_stat_chain[i, :] = stat_on_subsample

        sample_acceptance = accepted_samples / (i + 1)

        thetaNew = theta
        thetaP = theta + stepsize * npr.randn(2)
        u = npr.rand()
        lhds = getLogLhd(subsample, thetaP[0], np.exp(thetaP[1])) - getLogLhd(
            subsample, theta[0], np.exp(theta[1])
        )
        Lambda = np.mean(lhds)
        psi = 1.0 / N * np.log(u)
        if psi < Lambda:
            thetaNew = thetaP
            theta = thetaP
            accepted_theta += 1
            theta_chain[i, :] = thetaNew
        else:
            theta_chain[i, :] = theta

        theta_acceptance = accepted_theta / (i + 1)
        if np.mod(i, chain_length / 10) == 0:
            print(
                "Iteration",
                i,
                "Theta Acceptance",
                theta_acceptance,
                "Sample Acceptance",
                sample_acceptance,
            )

    return theta_chain, [], i
