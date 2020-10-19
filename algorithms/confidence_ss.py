import numpy as np
import numpy.random as npr
import time

# For the 'improved' confidence subsampler
import scipy.optimize as spo
import sympy as sym

from .settings import DEFAULT_MAX_CHAIN_LENGTH
from .settings import DEFAULT_TIMEOUT
from .utils import normalLogLhd

# Concentration bounds
def ctBernsteinSerfling(N, n, a, b, sigma, delta):
    """
    Bernstein-type bound without replacement, from (Bardenet and Maillard, to appear in Bernoulli)
    """
    l5 = np.log(5 / delta)
    kappa = 7.0 / 3 + 3 / np.sqrt(2)
    if n <= N / 2:
        rho = 1 - 1.0 * (n - 1) / N
    else:
        rho = (1 - 1.0 * n / N) * (1 + 1.0 / n)
    return sigma * np.sqrt(2 * rho * l5 / n) + kappa * (b - a) * l5 / n


def ctHoeffdingSerfling(N, n, a, b, delta):
    """
    Classical Hoeffding-type bound without replacement, from (Serfling, Annals of Stats 1974)
    """
    l2 = np.log(2 / delta)
    if n <= N / 2:
        rho = 1 - 1.0 * (n - 1) / N
    else:
        rho = (1 - 1.0 * n / N) * (1 + 1.0 / n)
    return (b - a) * np.sqrt(rho * l2 / 2 / n)


def ctBernstein(N, n, a, b, sigma, delta):
    """
    Classical Bernstein bound, see e.g. the book by Boucheron, Lugosi, and Massart, 2014.
    """
    l3 = np.log(3 / delta)
    return sigma * np.sqrt(2 * l3 / n) + 3 * (b - a) * l3 / n


def combineMeansAndSSQs(N1, mu1, ssq1, N2, mu2, ssq2):
    """
    combine means and sum of squares of two sets
    """
    dd = mu2 - mu1
    mu = mu1
    ssq = ssq1
    N = N1 + N2
    mu += dd * N2 / N
    ssq += ssq2
    ssq += (dd ** 2) * N1 * N2 / N
    return N, mu, ssq


# Differential functions for proxies,
# Define vectorized evaluation of gradient and Hessian
myGradientVect = lambda x_float, mu_float, sigma_float: np.array(
    [
        -(2 * mu_float - 2 * x_float) / (2 * sigma_float ** 2),
        -1 / sigma_float + (-mu_float + x_float) ** 2 / sigma_float ** 3,
    ]
).T
myHessianVect = lambda x_float, mu_float, sigma_float: [
    [
        -1 / sigma_float ** 2 * np.ones(x_float.shape),
        2 * (mu_float - x_float) / sigma_float ** 3,
    ],
    [
        2 * (mu_float - x_float) / sigma_float ** 3,
        (1 - 3 * (mu_float - x_float) ** 2 / sigma_float ** 2) / sigma_float ** 2,
    ],
]

# Compute third order derivatives to bound the Taylor remainder. Symbolic differentiation is not really necessary in this simple case, but
# it may be useful in later applications
def thirdDerivatives():
    x, mu, sigma = sym.symbols("x, mu, sigma")
    L = []
    for i in range(4):
        for j in range(4):
            if i + j == 3:
                args = tuple(
                    [-(x - mu) ** 2 / (2 * sigma ** 2) - sym.log(sigma)]
                    + [mu for cpt in range(i)]
                    + [sigma for cpt in range(j)]
                )
                L.append(sym.diff(*args))
    return L


def evalThirdDerivatives(x_float, mu_float, logSigma_float):
    tt = thirdDerivatives()
    return [
        tt[i]
        .subs("x", x_float)
        .subs("mu", mu_float)
        .subs("sigma", np.exp(logSigma_float))
        .evalf()
        for i in range(4)
    ]


# Confidence MCMC (Bardenet, Doucet, and Holmes, ICML'14)


def confidenceMCMC(
    initial_theta,
    x,
    delta=0.1,
    gamma=1.5,
    stepsize=0.01,
    time_budget=DEFAULT_TIMEOUT,
    chain_length=DEFAULT_MAX_CHAIN_LENGTH,
    getLogLhd=normalLogLhd,
):

    # Initialize
    N = len(x)
    theta = initial_theta
    S_B = np.zeros((chain_length, 2))
    acceptance = 0.0
    ns_B = []

    start_time = time.time()
    for i in range(chain_length):
        if time.time() - start_time > time_budget:
            print(
                f"Time budget consumed at chain step {i+1}, returning truncated result"
            )
            S_B = S_B[:i, :]
            break
        npr.shuffle(x)
        accepted = 0
        done = 0
        thetaNew = theta
        thetaP = theta + stepsize * npr.randn(2)
        u = npr.rand()
        n = N / 10
        cpt = 0
        lhds = getLogLhd(x, thetaP[0], np.exp(thetaP[1])) - getLogLhd(
            x, theta[0], np.exp(theta[1])
        )
        a = np.min(lhds)
        b = np.max(lhds)

        while not done and n < N:

            n = int(min(N, np.floor(gamma * n)))
            cpt += 1
            deltaP = delta / 2 / cpt ** 2
            # The following step should be done differently to avoid recomputing previous likelihoods, but for the toy examples we keep it short
            lhds = getLogLhd(x[:n], thetaP[0], np.exp(thetaP[1])) - getLogLhd(
                x[:n], theta[0], np.exp(theta[1])
            )
            Lambda = np.mean(lhds)
            sigma = np.std(lhds)
            psi = np.log(u) / N
            if np.abs(Lambda - psi) > ctBernstein(N, n, a, b, sigma, deltaP):
                done = 1

        if i > 1 and ns_B[-1] == 2 * N:
            ns_B.append(
                n
            )  # Half of the likelihoods were computed at the previous stage
        else:
            ns_B.append(
                2 * n
            )  # The algorithm required all likelihoods for theta and theta', next iteration we can reuse half of them

        if Lambda > psi:
            # Accept
            theta = thetaP
            accepted = 1
            S_B[i] = thetaP
        else:
            # Reject
            S_B[i] = theta

        if i < chain_length / 10:
            # Perform some adaptation of the stepsize in the early iterations
            stepsize *= np.exp(1.0 / (i + 1) ** 0.6 * (accepted - 0.5))

        acceptance *= i
        acceptance += accepted
        acceptance /= i + 1
        if np.mod(i, chain_length / 10) == 0:
            # Monitor acceptance and average number of samples used
            print(
                "Iteration",
                i,
                "Acceptance",
                acceptance,
                "Avg. num evals",
                np.mean(ns_B),
                "sigma/sqrt(n)",
                sigma / np.sqrt(n),
                "R/n",
                (b - a) / n,
            )

    return S_B, []


# Confidence MCMC with proxy (Bardenet, Doucet, and Holmes, this submission)
def confidenceMCMCWithProxy(
    initial_theta,
    x,
    delta=0.1,
    gamma=1.5,
    stepsize=0.1,
    time_budget=DEFAULT_TIMEOUT,
    chain_length=DEFAULT_MAX_CHAIN_LENGTH,
    getLogLhd=normalLogLhd,
):
    # Find the MAP (not really necessary here since the MAP are the mean and std deviation of the data)
    f = lambda theta: -np.mean(getLogLhd(x, theta[0], np.exp(theta[1])))
    thetaMAP = spo.minimize(f, initial_theta).x
    print("MAP is", thetaMAP, "Real values are", initial_theta[0], initial_theta[1])
    tt = thirdDerivatives()
    print(tt)

    npr.seed(1)
    # Initialize
    N = len(x)
    theta = initial_theta
    S_B = np.zeros((chain_length, 2))
    acceptance = 0.0
    ns_B = []

    # Compute some statistics of the data that will be useful for bounding the error and averaging the proxies
    minx = np.min(x)
    maxx = np.max(x)
    meanx = np.mean(x)
    meanxSquared = np.mean(x ** 2)

    # Prepare total sum of Taylor proxys
    muMAP = thetaMAP[0]
    sigmaMAP = np.exp(thetaMAP[1])
    meanGradMAP = np.array(
        [
            (meanx - muMAP) / sigmaMAP ** 2,
            (meanxSquared - 2 * muMAP * meanx + muMAP ** 2) / sigmaMAP ** 3
            - 1.0 / sigmaMAP,
        ]
    )
    meanHessMAP = np.array(
        [
            [-1.0 / sigmaMAP ** 2, -2 * (meanx - muMAP) / sigmaMAP ** 3],
            [
                -2 * (meanx - muMAP) / sigmaMAP ** 3,
                -3 * (meanxSquared - 2 * muMAP * meanx + muMAP ** 2) / sigmaMAP ** 4
                + 1 / sigmaMAP ** 2,
            ],
        ]
    )

    start_time = time.time()
    for i in range(chain_length):
        if time.time() - start_time > time_budget:
            print(
                f"Time budget consumed at chain step {i+1}, returning truncated result"
            )
            S_B = S_B[:i, :]
            break
        npr.shuffle(x)
        accepted = 0
        done = 0
        thetaNew = theta
        thetaP = theta + stepsize * npr.randn(2)
        u = npr.rand()
        n = 2
        t0 = 0
        cpt = 0
        Lambda = 0
        ssq = 0  # Sum of squares

        # Prepare Taylor bounds
        xMinusMuMax = np.max(
            np.abs(
                [
                    1,
                    minx - theta[0],
                    maxx - theta[0],
                    minx - thetaMAP[0],
                    maxx - thetaMAP[0],
                    minx - thetaP[0],
                    maxx - thetaP[0],
                ]
            )
        )
        sigmaMin = np.min(np.exp([theta[1], thetaMAP[1], thetaP[1]]))
        R = float(
            np.max(np.abs(evalThirdDerivatives(xMinusMuMax, 0, np.log(sigmaMin))))
        )
        h = np.array([theta[0] - thetaMAP[0], np.exp(theta[1]) - np.exp(thetaMAP[1])])
        hP = np.array(
            [thetaP[0] - thetaMAP[0], np.exp(thetaP[1]) - np.exp(thetaMAP[1])]
        )
        R *= 2 * 1.0 / 6 * max(np.sum(np.abs(h)), np.sum(np.abs(hP))) ** 3
        a = -R
        b = R

        # We can already compute the average proxy log likelihood ratio
        avgTotalProxy = np.dot(meanGradMAP, hP - h) + 0.5 * np.dot(
            hP - h, np.dot(meanHessMAP, h + hP)
        )

        while not done and n < N:

            n = int(min(N, np.floor(gamma * n)))
            cpt += 1
            deltaP = delta / 2 / cpt ** 2
            batch = x[t0:n]
            lhds = getLogLhd(batch, thetaP[0], np.exp(thetaP[1])) - getLogLhd(
                batch, theta[0], np.exp(theta[1])
            )
            proxys = np.dot(
                myGradientVect(batch, muMAP, sigmaMAP), hP - h
            ) + 0.5 * np.dot(
                np.dot(hP - h, myHessianVect(batch, muMAP, sigmaMAP)).T, h + hP
            )
            if np.any(np.abs(lhds - proxys) > R):
                # Just a check that our error is correctly bounded
                print("Taylor remainder is underestimated")
            _, Lambda, ssq = combineMeansAndSSQs(
                t0,
                Lambda,
                ssq,
                n - t0,
                np.mean(lhds - proxys),
                (n - t0) * np.var(lhds - proxys),
            )
            sigma = np.sqrt(1.0 / n * ssq)
            psi = np.log(u) / N
            t0 = n
            if np.abs(Lambda - psi + avgTotalProxy) > ctBernstein(
                N, n, a, b, sigma, deltaP
            ):
                done = 1

        if i > 1 and ns_B[-1] == 2 * N:
            ns_B.append(
                n
            )  # Half of the likelihoods were computed at the previous stage
        else:
            ns_B.append(2 * n)

        if Lambda + avgTotalProxy > psi:
            # Accept
            theta = thetaP
            accepted = 1
            S_B[i] = thetaP
        else:
            # Reject
            S_B[i] = theta

        if i < chain_length / 10:
            # Perform some adaptation of the stepsize in the early iterations
            stepsize *= np.exp(1.0 / (i + 1) ** 0.6 * (accepted - 0.5))

        acceptance *= i
        acceptance += accepted
        acceptance /= i + 1
        if np.mod(i, chain_length / 10) == 0:
            # Monitor acceptance and average number of samples used
            print(
                "Iteration",
                i,
                "Acceptance",
                acceptance,
                "Avg. num samples",
                np.mean(ns_B),
                "Dist. to MAP",
                np.sum(np.abs(theta - thetaMAP)),
                "sigma/sqrt(n)",
                sigma / np.sqrt(n),
                "R/n",
                R / n,
            )

    return S_B, ns_B


def confidenceMCMCWithProxyDroppedAlong(
    initial_theta,
    x,
    delta=0.1,
    gamma=1.5,
    stepsize=0.1,
    time_budget=DEFAULT_TIMEOUT,
    chain_length=DEFAULT_MAX_CHAIN_LENGTH,
    getLogLhd=normalLogLhd,
):
    """
    perform confidence MCMC with proxy dropped every 20 iterations
    """

    def dropProxy(thetaStar, meanx, minx, maxx, meanxSquared):
        """
        compute all quantities necessary to the evaluation of a proxy at thetaStar
        """
        muStar = thetaStar[0]
        sigmaStar = np.exp(thetaStar[1])
        meanGradStar = np.array(
            [
                (meanx - muStar) / sigmaStar ** 2,
                (meanxSquared - 2 * muStar * meanx + muStar ** 2) / sigmaStar ** 3
                - 1.0 / sigmaStar,
            ]
        )
        meanHessStar = np.array(
            [
                [-1.0 / sigmaStar ** 2, -2 * (meanx - muStar) / sigmaStar ** 3],
                [
                    -2 * (meanx - muStar) / sigmaStar ** 3,
                    -3
                    * (meanxSquared - 2 * muStar * meanx + muStar ** 2)
                    / sigmaStar ** 4
                    + 1 / sigmaStar ** 2,
                ],
            ]
        )
        return meanGradStar, meanHessStar

    npr.seed(1)
    # Initialize
    N = len(x)
    # Find the MAP (not really necessary here since the MAP are the mean and std deviation of the data)
    f = lambda theta: -np.mean(getLogLhd(x, theta[0], np.exp(theta[1])))
    thetaMAP = spo.minimize(f, initial_theta).x
    theta = initial_theta
    S_B = np.zeros((chain_length, 2))
    acceptance = 0.0
    ns_B = []

    # Compute min and max of data
    minx = np.min(x)
    maxx = np.max(x)
    meanx = np.mean(x)
    meanxSquared = np.mean(x ** 2)

    # Prepare Taylor proxys
    thetaStar = thetaMAP
    muStar = thetaStar[0]
    sigmaStar = np.exp(thetaStar[1])
    meanGradStar, meanHessStar = dropProxy(thetaStar, meanx, minx, maxx, meanxSquared)

    start_time = time.time()
    for i in range(chain_length):
        if time.time() - start_time > time_budget:
            print(
                f"Time budget consumed at chain step {i+1}, returning truncated result"
            )
            S_B = S_B[:i, :]
            break
        npr.shuffle(x)
        accepted = 0
        done = 0
        thetaNew = theta
        thetaP = theta + stepsize * npr.randn(2)
        u = npr.rand()
        n = 2
        t0 = 0
        cpt = 0
        Lambda = 0
        ssq = 0

        # Prepare Taylor bounds
        xMinusMuMax = np.max(
            np.abs(
                [
                    1,
                    minx - theta[0],
                    maxx - theta[0],
                    minx - thetaStar[0],
                    maxx - thetaStar[0],
                    minx - thetaP[0],
                    maxx - thetaP[0],
                ]
            )
        )
        sigmaMin = np.min(np.exp([theta[1], thetaStar[1], thetaP[1]]))
        R = float(
            np.max(np.abs(evalThirdDerivatives(xMinusMuMax, 0, np.log(sigmaMin))))
        )
        h = np.array([theta[0] - thetaStar[0], np.exp(theta[1]) - np.exp(thetaStar[1])])
        hP = np.array(
            [thetaP[0] - thetaStar[0], np.exp(thetaP[1]) - np.exp(thetaStar[1])]
        )
        R *= 2 * 1.0 / 6 * max(np.sum(np.abs(h)), np.sum(np.abs(hP))) ** 3
        a = -R
        b = R

        avgTotalProxy = np.dot(meanGradStar, hP - h) + 0.5 * np.dot(
            hP - h, np.dot(meanHessStar, h + hP)
        )

        while not done and n < N:

            n = int(min(N, np.floor(gamma * n)))

            if not np.mod(
                i, 20
            ):  # Loop over whole dataset and recompute proxys when finished
                n = N

            cpt += 1
            deltaP = delta / 2 / cpt ** 2
            batch = x[t0:n]
            lhds = getLogLhd(batch, thetaP[0], np.exp(thetaP[1])) - getLogLhd(
                batch, theta[0], np.exp(theta[1])
            )
            proxys = np.dot(
                myGradientVect(batch, muStar, sigmaStar), hP - h
            ) + 0.5 * np.dot(
                np.dot(hP - h, myHessianVect(batch, muStar, sigmaStar)).T, h + hP
            )
            if np.any(np.abs(lhds - proxys) > R):
                print("Taylor remainder is underestimated")
            _, Lambda, ssq = combineMeansAndSSQs(
                t0,
                Lambda,
                ssq,
                n - t0,
                np.mean(lhds - proxys),
                (n - t0) * np.var(lhds - proxys),
            )
            sigma = np.sqrt(1.0 / n * ssq)
            psi = np.log(u) / N
            t0 = n
            # print "n, abs(L-psi), bound, sigma/sqrt(n), R/n", n, np.abs(Lambda-psi), ctBernstein(N,n,a,b,sigma,deltaP), sigma/np.sqrt(n), R/n
            if np.abs(Lambda - psi + avgTotalProxy) > ctBernstein(
                N, n, a, b, sigma, deltaP
            ):
                done = 1

        if i > 1 and ns_B[-1] == 2 * N:
            ns_B.append(
                n
            )  # Half of the likelihoods were computed at the previous stage
        else:
            ns_B.append(2 * n)

        if not np.mod(i, 20):  # Recompute proxys every 20 iterations
            thetaStar = theta
            muStar = thetaStar[0]
            sigmaStar = np.exp(thetaStar[1])
            meanGradStar, meanHessStar = dropProxy(
                thetaStar, meanx, minx, maxx, meanxSquared
            )

        if Lambda + avgTotalProxy > psi:
            # Accept
            theta = thetaP
            accepted = 1
            S_B[i] = thetaP
        else:
            # Reject
            S_B[i] = theta

        if i < chain_length / 10:
            # Perform some adaptation of the stepsize in the early iterations
            stepsize *= np.exp(1.0 / (i + 1) ** 0.6 * (accepted - 0.5))

        acceptance *= i
        acceptance += accepted
        acceptance /= i + 1
        if np.mod(i, chain_length / 10) == 0:
            # Monitor acceptance and average number of samples used
            print(
                "Iteration",
                i,
                "Acceptance",
                acceptance,
                "Avg. num samples",
                np.mean(ns_B),
                "Dist. to MAP",
                np.sum(np.abs(theta - thetaMAP)),
                "sigma/sqrt(n)",
                sigma / np.sqrt(n),
                "R/n",
                R / n,
            )

    return S_B, ns_B
