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

for n in [10, 50]:

    x = np.random.multivariate_normal(mu, sigma, size=n)
    p=x.shape[1]

    xbar = x.mean(axis=0)
    s = np.cov(x.T)
    s_inv = np.linalg.inv(s)
    T2 = n*xbar@s_inv@xbar

    f_val=f.ppf(1-alpha, p, n-p)
    t2_crit = (p*(n-1)/(n-p))*f_val
    print(f' n={n}\ncritical t2 {t2_crit}\ncalculated t2 {T2}')

"""
The file two sample-data.csv contains a sample of 3-variate normal data
data Xj = (Xj1 , Xj2 , Xj3 )′ (with variables v1, v2 and v3), j = 1, . . . , 150,
and a categorial variable type with levels 1 and 2. We would like to test if
there is a significant difference in the mean of those variables with regard
to their type.
    (a) Does it seem reasonable to assume that the covariance matrices under
        type 1 and 2 are identical?
    (b) Use Hotelling’s two-sample test to test H0 : µ1 = µ2 vs. HA : µ1 ̸= µ2 ,
        where µi = E[Xj |type = i], i = 1, 2 at significance level α = 0.05.
        Provide the p-value.
"""
# a) yes, because both should be the ones for bivariate normal
#     however, there are 2x as many samples for type 2, so that matrix is likely closer to the true one

# b)
X = pd.read_csv('/home/paul/Desktop/cosi_importanti/uni/DS/sem_2/applied_stat/Data-20261011/two_sample-data.csv')
X1 = X[X.type==1]
X2 = X[X.type==2]

mu1 = X1.mean(axis=0)[1:]
mu2 = X2.mean(axis=0)[1:]
alpha=0.05
n1 = X[X.type==1].count().type
n2 = X[X.type==2].count().type

Sp = ((n1 - 1) * np.cov(X1.iloc[:, 1:4].T) + (n2 - 1) * np.cov(X2.iloc[:, 1:4].T)) / (n1 + n2 - 2)
diff = mu1.values - mu2.values
T2 = (n1 * n2) / (n1 + n2) * diff @ np.linalg.inv(Sp) @ diff

p = 3
F_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * T2
F_crit = f.ppf(1 - alpha, p, n1 + n2 - p - 1)
print(f'\n\n\nex6\nF_calc: {F_stat}\nF_threshold: {F_crit}')
