#exercise one
"""
1. Analyse the iris data in R (iris is a toy data frame in R). The iris data
set gives the measurements in centimeters of the variables sepal length and
width and petal length and width, respectively, for 50 flowers from each
of 3 species of iris. The species are Iris setosa, versicolor, and virginica.
    (a) Determine mean and correlation matrix and make a scatter plot matrix ignoring the species.
    (b) Repeat the exercise, but now distinguish between the different species.
       What is your conclusion?
    (c) Choose the species setosa and plot sepal.length v.s. sepal.width. Do
       the data look approximately bivariate normal? Plot the data along
       with a 95% tolerance zone (based on the empirical mean and the
       empirical covariance matrix).
    (d) Consider again the species setosa and consider now all four variables
       sepal.length, sepal.width, petal.length and petal.width. Are there
       any data points not contained in the 95% tolerance zone? If yes,
       mark them in the scatter plot matrix.
"""
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2, kde

raw = datasets.load_iris()
iris = raw.data
setosa = iris[:50]
virginica = iris[50:100]
versicolor = iris[100:]

def get_basic_info(dataset, plotname):
    means = [dataset[:,i].mean() for i in range(4)]
    df = pd.DataFrame(dataset, columns=raw.feature_names)
    corr = df.corr()
    print(f'means: {means} \n corr: {corr}')
    sm = pd.plotting.scatter_matrix(df, diagonal='kde')
    plt.savefig(plotname)
    plt.cla()
    return [means, corr, df]

iris_d = get_basic_info(iris, 'iris.png')
setosa_d = get_basic_info(setosa, 'setosa.png')
virginica_d = get_basic_info(virginica, 'virginica.png')
versicolor_d = get_basic_info(versicolor, 'versicolor.png')

data = setosa[:, :2]
mu = data.mean(axis=0)
cov = np.cov(data.T)

vals, vecs = np.linalg.eigh(cov)
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
width, height = 2 * np.sqrt(chi2.ppf(0.95, 2) * vals)

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(mu[0], mu[1], c='red')
ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
              edgecolor='red', facecolor='none', lw=2)
plt.gca().add_patch(ell)
plt.xlabel('sepal length'); plt.ylabel('sepal width')
plt.title('Setosa: sepal length vs sepal width (95% tolerance)')
plt.savefig('setosa_sepal.png')
plt.show()

data = setosa
mu = data.mean(axis=0)
cov = np.cov(data.T)
inv_cov = np.linalg.inv(cov)

diff = data -mu
d2 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
threshold = chi2.ppf(0.95, df=4)
outliers = d2 > threshold
print("outlier indices:", np.where(outliers)[0])
print("num outliers:", outliers.sum())
df = pd.DataFrame(data, columns=raw.feature_names)
sm = pd.plotting.scatter_matrix(df, figsize=(8, 8), diagonal='hist')
for i in range(4):
    for j in range(4):
        if i != j:
            sm[i, j].scatter(data[outliers, j], data[outliers, i],
                             c='red', s=60, edgecolors='black')
plt.suptitle('Setosa: outliers (red) outside 95% tolerance zone')
plt.savefig('setosa_outliers.png')
plt.show()
