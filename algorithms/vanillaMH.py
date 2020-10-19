import numpy as np
import numpy.random as npr
import time

from .settings import DEFAULT_MAX_CHAIN_LENGTH
from .settings import DEFAULT_TIMEOUT
from .utils import normalLogLhd


def vanillaMH(
    inital_theta,
    x,
    stepsize=0.5,
    time_budget=DEFAULT_TIMEOUT,
    chain_length=DEFAULT_MAX_CHAIN_LENGTH,
    getLogLhd=normalLogLhd,
):
    """
    Perform traditional isotropic random walk Metropolis

    We discard the initial value in every chain.

    :param initial_theta: The initial array representing the start of the Markov Chain for theta 
    :param x: The data on which you are trying to fit a posterior with a likelihood
            of drawn data given by :param getLogLhd: - which defaults to a normal likelihood model.
    :param time_budget: Optional. How long we're allowing the chain to run for in seconds. 
            Default 10mins.
    :param chain_length: Optional. Maximum chain length - used to reserve memory.
            Defaults 10k.
    :param getLogLhd: Optional. The function for computing log-likelihood of an iid observation.
            Default normal log likelihood.
    """
    N = len(x)
    theta = inital_theta
    stepsize = stepsize / np.sqrt(N)
    S = np.zeros((chain_length, 2))
    acceptance = 0.0
    accepted = 0

    start_time = time.time()
    for i in range(chain_length):
        if time.time() - start_time > time_budget:
            print(
                f"Time budget consumed at chain step {i+1}, returning truncated result"
            )
            S = S[:i, :]
            break
        stepsize_adapt = 0
        thetaNew = theta
        thetaP = theta + stepsize * npr.randn(2)
        u = npr.rand()
        lhds = getLogLhd(x, thetaP[0], np.exp(thetaP[1])) - getLogLhd(
            x, theta[0], np.exp(theta[1])
        )
        Lambda = np.mean(lhds)
        psi = 1.0 / N * np.log(u)
        if psi < Lambda:
            thetaNew = thetaP
            theta = thetaP
            accepted += 1
            stepsize_adapt = 1
            S[i, :] = thetaNew
        else:
            S[i, :] = theta

        if i < chain_length / 10:
            # Perform some adaptation of the stepsize in the early iterations
            stepsize *= np.exp(1.0 / (i + 1) ** 0.6 * (stepsize_adapt - 0.5))

        acceptance = accepted / (i + 1)
        if np.mod(i, chain_length / 10) == 0:
            print("Iteration", i, "Acceptance", acceptance)

    return S

