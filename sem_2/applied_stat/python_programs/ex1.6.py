
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
# a) yes, because both covariance matrices should be the ones for bivariate normal
#     however, there are more samples for type 2, so that matrix is likely closer to the one achieved at  n -> infty
#     also, if the level selection was made after the generation of data, whether we have the same covariance deppends on if this
#     selection was made at random or according to some given criteria

# b)
import pandas as pd
import numpy as np
from scipy.stats import norm, f
import matplotlib.pyplot as plt

X = pd.read_csv('/home/paul/Desktop/cosi_importanti/uni/DS/sem_2/applied_stat/Data-20261011/two_sample-data.csv')
X1 = X[X.type==1]
X2 = X[X.type==2]

mu1 = X1.mean(axis=0)[1:]
mu2 = X2.mean(axis=0)[1:]
alpha=0.05
n1 = len(X1)
n2 = len(X2)

xax1  =np.linspace(0, n1, num=n1)
xax2  =np.linspace(0, n2, num=n2)

fig, axs = plt.subplots(3,3)
axs[0, 1].scatter(X1.iloc[:, 1], X1.iloc[:, 2], c='green')
axs[1, 0].scatter(X1.iloc[:, 1], X1.iloc[:, 2], c='green')
axs[0, 1].scatter(X2.iloc[:, 1], X2.iloc[:, 2], c='red')
axs[1, 0].scatter(X2.iloc[:, 1], X2.iloc[:, 2], c='red')
#... and so on

#unbiased pooled covariance
Sp = ((n1 - 1) * np.cov(X1.iloc[:, 1:4].T) + (n2 - 1) * np.cov(X2.iloc[:, 1:4].T)) / (n1 + n2 - 2)
diff = mu1.values - mu2.values

T2 = (n1 * n2) / (n1 + n2) * diff @ np.linalg.inv(Sp) @ diff

p = len(X1.iloc[0]) -1
F_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * T2
F_crit = f.ppf(1 - alpha, p, n1 + n2 - p - 1)
pval= 1 - f.cdf(F_stat, p, n1+n1-p-1)
print(f'\n\n\nex6\nF_calc: {F_stat}\nF_crit: {F_crit}\n p: {pval}')
