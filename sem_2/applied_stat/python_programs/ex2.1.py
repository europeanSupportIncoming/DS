"""
(Exercise 8.10 of J&W) The data set T8-4.dat contains weekly rates of
return for five stocks listed on the New York Stock Exchange. The first
three stocks are banking stocks the others are energy stocks.
(a) Determine the sample covariance matrix S and find the sample prin-
cipal components.
(b) Determine the proportion of the total sample variance explained by
the first three principal components. Interpret these components.
(c) Do you think the stock rates-of-return data can be summarized in
fewer than five dimensions?
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = pd.read_csv('/home/paul/Desktop/cosi_importanti/uni/DS/sem_2/applied_stat/Data-20261011/T8-4.DAT', delimiter="\t")
S = X.cov()
eigs  = np.linalg.eigh(S)
normed_eigvals = eigs.eigenvalues/np.linalg.norm(S)
#            0.0130338  -0.0078431  -0.0031889  -0.0447693  0.0052151
# 0.0130338    0.000436    0.000279    0.000161    0.000071   0.000090
# -0.0078431   0.000279    0.000442    0.000181    0.000179   0.000125
# -0.0031889   0.000161    0.000181    0.000226    0.000072   0.000061
# -0.0447693   0.000071    0.000179    0.000072    0.000706   0.000514
# 0.0052151    0.000090    0.000125    0.000061    0.000514   0.000773

pca = PCA()
pca.fit(X)
print(pca.components_)
# array([[ 0.22952328,  0.30881581,  0.15570887,  0.63041982,  0.65595538],
#        [ 0.62077314,  0.57050563,  0.34460556, -0.24453285, -0.33258792],
#        [-0.3359429 ,  0.2580364 ,  0.02687116,  0.64765674, -0.63275451],
#        [ 0.66029623, -0.41073056, -0.49951251,  0.31248823, -0.21942576],
#        [ 0.11450341, -0.58640749,  0.77895247,  0.16020734, -0.10286826]])

print(pca.explained_variance_)
# array([0.00137169, 0.00070652, 0.00024148, 0.00014399, 0.00011981])

print(pca.explained_variance_ratio_)
# array([0.53094492, 0.2734745 , 0.0934698 , 0.05573439, 0.04637639])

print(np.cumsum(pca.explained_variance_ratio_))
# array([0.53094492, 0.80441942, 0.89788922, 0.95362361, 1.        ])
# --> 89 %
plt.imshow(S)
plt.colorbar()
plt.show()


