"""
In this exercise we want to empirically explore the power of Hotelling’s T 2
test, by a Monte Carlo simulation. We consider bivariate normal data and
test H0 : µ = (0, 0) at significance level α. Then Hotelling’s T 2 statistic
for a sample of size n is
T 2 = nX̄ ′ S −1 X̄.
We consider sample sizes (a) n = 10 and (b) n = 50. The critical region
is K = {x : nx̄′ s−1 x̄ ≥ κ1−α }. Here κ1−α is the (1 − α)-quantile of the
corresponding Hotelling T 2 distribution and x = (x′1 , . . . , x′n )′ with xi ∈
R2 . Recall that the power function for the test is
β(µ, Σ) = Pµ,Σ (T 2 > κ1−α ).
This is the probability that T 2 exceeds the critical value, given the true
parameters are µ, Σ.
    (a) Think of how we can approximate β(µ, Σ) for given values of µ and
        Σ by a simulation experiment (Monte Carlo simulation).
    (b) Can you argue why your method would work in theory?
    (c) Implement your method. In case you use R you can use the package
        mvtnorm to generate i.i.d. normal data. Use α = 0.05. What are your
        findings?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random.mtrand import normal
from scipy.stats import f


# a): MC sim -> make n points according to bivariate dist
# get sigma as cov of these
# calc n samples w. numpy multivariate_normal -> get T²
# calc sigma
# get nr above kappa²/total n

# b) LOLn

# c)
mu = [0.0, 0.0]
alpha=0.05

A=np.random.rand(2,2)
sigma = A@A.T + 0.1*np.eye(2)

print(f'sigma {sigma}')
n_reps = 10000
successes = np.zeros(n_reps)
for repeat in range(n_reps):
    for n in [10, 50]:

        x = np.random.multivariate_normal(mu, sigma, size=n)
        p=x.shape[1]

        xbar = x.mean(axis=0)
        s = np.cov(x.T)
        s_inv = np.linalg.inv(s)
        T2 = n*xbar@s_inv@xbar

        f_val=f.ppf(1-alpha, p, n-p)
        t2_crit = (p*(n-1)/(n-p))*f_val
        # print(f' n={n}\ncritical t2 {t2_crit}\ncalculated t2 {T2}')
    successes[repeat] = T2 <= t2_crit
print(np.sum(successes)/len(successes))
