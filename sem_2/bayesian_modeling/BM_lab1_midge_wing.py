import numpy as np
from pandas import plotting
from scipy.stats import multivariate_normal, invgamma
import matplotlib.pyplot as plt

x = np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.90, 2.08])
n = len(x)
m0 = 1.9
M0 = 1
nu0 = 2
sigsq0 = 0.01


def predict_sigsq(M0=1, m0=1.0, x=np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.90, 2.08]), nu0=2, sigsq0=0.01, plotting=False):
    n = len(x)
    xbar = np.average(x)
    Mn = M0 / (M0 * n + 1)
    mn = Mn * (m0 / M0 + xbar)
    nun = nu0 + n
    sigsqn = 1 / nun * (nu0 * sigsq0 ** 2 + np.linalg.norm(x - xbar) ** 2 + np.linalg.norm(xbar - m0) ** 2 / (M0 + 1 / n))
    post_sig = invgamma(nun/2, sigsqn*nun/2)


    post_mu = np.random.normal(mn, Mn*sigsqn)
    mu = (1/M0)/(1/M0 + n)*m0 + n/(1/M0 + n)*xbar
    res = sigsqn * nun / (nun - 2)





predict_sigsq()
for M,m in np.random.random((10, 2)):
    predict_sigsq(M0=M, m0=m)


