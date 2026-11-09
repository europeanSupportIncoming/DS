"""
Consider the data set T1-5.dat. The components are x1 = wind, x2 =
solar radiation, x3 = CO, x4 = NO, x5 = NO2 , x6 = O3 and x7 = HC.
Your job is to summarize these data in fewer than p = 7 dimensions if
possible.
(a) Conduct a principal component analysis of the data using both the
covariance and the correlation matrix and explain your results.
(b) Can you interpret the principal components?
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = pd.read_csv('/home/paul/Desktop/cosi_importanti/uni/DS/sem_2/applied_stat/Data-20261011/T1-5.dat', delimiter=r"\s", dtype=float)
S = X.cov()
eigs  = np.linalg.eigh(S)
normed_eigvals = eigs.eigenvalues/np.linalg.norm(S)

pca = PCA()
pca.fit(X)
pca.components_
# array([[-1.16212791e-02,  9.92204334e-01,  9.75022579e-03,
#         -4.58753338e-03,  2.13777050e-02,  1.21655993e-01,
#          4.71234561e-03],
#        [-7.46006944e-02, -1.24482417e-01,  1.09170074e-01,
#         -1.28570942e-02,  1.61916627e-01,  9.69647875e-01,
#          2.05294664e-02],
#        [-3.32162324e-02, -2.20098952e-04,  1.69835228e-01,
#          1.32397701e-01,  9.54585444e-01, -1.81294025e-01,
#          9.17746319e-02],
#        [ 9.18983962e-01,  1.84876843e-04, -1.43513543e-01,
#         -3.27882380e-01,  1.04124508e-01,  6.27746179e-02,
#          1.12173459e-01],
#        [ 3.46185044e-01,  5.10492881e-03,  5.98207862e-01,
#          6.84533305e-01, -2.00487842e-01,  3.42312960e-05,
#          1.16172229e-01],
#        [-2.94478975e-02, -2.60055283e-03, -5.98225252e-01,
#          4.20283448e-01, -7.32371877e-03,  5.71690885e-02,
#          6.79185165e-01],
#        [-1.67193235e-01, -1.33479208e-03,  4.72038757e-01,
#         -4.79117543e-01, -1.04889009e-01, -7.00452420e-02,
#          7.09762208e-01]])

pca.explained_variance_
# array([2.97274539e+02, 2.86194297e+01, 1.16129439e+01, 2.58671450e+00,
#        1.27639156e+00, 4.02533623e-01, 2.14032706e-01])

pca.explained_variance_ratio_
# array([8.69257895e-01, 8.36858254e-02, 3.39573081e-02, 7.56378938e-03,
#        3.73228546e-03, 1.17704507e-03, 6.25851175e-04])

np.cumsum(pca.explained_variance_ratio_)
# array([0.8692579 , 0.95294372, 0.98690103, 0.99446482, 0.9981971 ,
#        0.99937415, 1.        ])

