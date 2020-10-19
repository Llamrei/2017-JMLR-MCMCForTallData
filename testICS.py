from algorithms.confidence_ss import confidenceMCMC

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.stats as sps
import seaborn as sns
import seaborn.distributions as snsd

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
ics_chain, _ = confidenceMCMC(initial_theta, x)
