# MCMC for tall data

Python 3.7.9 code to reproduce the Metroplis-Hastings (MH) and Improved Confidence Sampler (ICS) toy examples in [(Bardenet, Doucet, and Holmes, 2017)](https://arxiv.org/abs/1505.02827) and extend the toy examples to compare and contrast to the Informed Sub-Sampling (ISS) method proposed in [(Maire, F., Friel, N. and Alquier, P., 2019.)](https://arxiv.org/abs/1706.08327).

For ISS we use a guesstimated symmetric proposal function based on the uniform swapping of elements in and out of the subsample.

For a static html rendering of the code, just click on the .ipynb.

More complete text references of aforementioned papers for link stability issues:
- Bardenet, R., Doucet, A. and Holmes, C., 2017. On Markov chain Monte Carlo methods for tall data. The Journal of Machine Learning Research, 18(1), pp.1515-1557.
- Maire, F., Friel, N. and Alquier, P., 2019. Informed sub-sampling MCMC: approximate Bayesian inference for large datasets. Statistics and Computing, 29(3), pp.449-482.