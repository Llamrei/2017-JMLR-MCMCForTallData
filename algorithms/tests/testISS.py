from ..iss import iss_mcmc


#  Uncomment to test with: python3 .\algorithms\iss.py
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import scipy.special as spsp
import scipy.misc as spm
import scipy.optimize as spo
import numpy.linalg as npl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import random
import sympy as sym
import time
import seaborn as sns
import seaborn.distributions as snsd
import math as math

# Generate data
npr.seed(1)
N = 100000
# Here is where we make the model mis-specified
dataType = "Gaussian"
x = npr.randn(N)

# We store the mean and std deviation for later reference, they are also the MAP and MLE estimates in this case.
realMean = np.mean(x)
realStd = np.std(x)
print("Mean of x =", realMean)
print("Std of x =", realStd)

# Where we will start all theta chains
initial_theta = np.array([realMean, np.log(realStd)])
iss_chain, _ = iss_mcmc(initial_theta, x)
