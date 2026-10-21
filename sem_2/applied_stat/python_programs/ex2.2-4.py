"""
2. The file mvnorm.csv contains 50 data points from a three-dimensional
distribution.
    (a) Estimate µ and Σ.
    (b) Make a scatterplot matrix of the data. Do the data look multivariate
        normal?
    (c) Assuming the data are normal, calculate (explicitly) 95%-tolerance
        zones and check which, if any, data are outside the tolerance zone.
        ˆ︁ −1/2 (Xi − X̄). Consider again the
    (d) Standardise the data to Yi = Σ
        scatterplots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Ellipse
from scipy.stats import chi2, kde, f, t


mvnorm = pd.read_csv('/home/paul/Desktop/cosi_importanti/uni/DS/sem_2/applied_stat/Data-20261011/mvnorm.csv', index_col=0)
X = mvnorm.values
mu = X.mean(axis=0)
sigma = np.cov(X.T, ddof=1)
inv_sigma = np.linalg.inv(sigma)

print(f'\n\nmu {mu} \n\n sigma {sigma}\n\n')
# pd.plotting.scatter_matrix(mvnorm, diagonal='kde')
# plt.show()
# plt.savefig('scatter_raw_ex2.png')

diff = X-mu
d2 = np.einsum('ij,jk,ik->i', diff, inv_sigma, diff)
threshold = chi2.ppf(0.95, df=3)

outsiders = d2 > threshold
print(f'outside points: {mvnorm[outsiders]}')

vals, vecs = np.linalg.eigh(sigma)
sigma_inv_sqrt = vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T

Y = (X - mu) @ sigma_inv_sqrt.T
Y_df = pd.DataFrame(Y, columns=['Y1', 'Y2', 'Y3'])
# pd.plotting.scatter_matrix(Y_df, diagonal='kde')
# plt.show()
# plt.savefig('Yi_normalized_ex2.png')

"""
In the previous exercise:
    (a) Assuming the data are normal, calculate (explicitly) 95%-rectangular
        conference zone (by Bonferroni method).
    (b) Check the Hypothesis H0 : µ = µ0 = (0.8, 2.1, 1.7)′ vs. HA : µ ̸= µ0 .
    (c) From historical data we know
        Σ = [
        [4.95, 0.99, 0.07],
        [0.99, 0.50, 0.02],
        [0.07, 0.02, 0.999]
        ]
        Plot a control chart with Hotelling’s T 2 statistics for batches of 10
        consecutive data.
"""

n, p = X.shape
s = X.std(axis=0, ddof=1)
tq = t.ppf(1 - 0.05 / (2*p), df=n-1)
half = tq*s/np.sqrt(n)

lower = mu - half
upper = mu + half

print(f'lower:{lower}\n upper:{upper}')
mu_0 = np.array([0.8, 2.1, 1.7]).T

diff_h0 = mu - mu_0
T2 = n*diff_h0 @ inv_sigma @ diff_h0
F_stat = (n - p) / ((n - 1) * p) * T2
p_value = 1 - f.cdf(F_stat, p, n - p)
F_crit = f.ppf(0.95, p, n - p)

print(f'F_crit:{F_crit:.4f}\n p_value:{p_value:.4f} \n F_stat:{F_stat:.4f} \n T2:{T2:.4f}\n')

sigma_hist = np.array([
[4.95, 0.99, 0.07],
[0.99, 0.50, 0.02],
[0.07, 0.02, 0.999]])

m = 10
n_batches = n // m
batch_means = X[:n_batches * m].reshape(n_batches, m, p).mean(axis=1)

inv_sigma_hist = np.linalg.inv(sigma_hist)
diff_b = batch_means - mu
T2_batches = m * np.einsum('ij,jk,ik->i', diff_b, inv_sigma_hist, diff_b)

ucl = (p * (m - 1) / (m - p)) * f.ppf(0.95, p, m - p)
print(f'UCL: {ucl:.4f}\n T2 batches: {np.round(T2_batches, 4)}')

# plt.figure()
# plt.plot(range(1, n_batches + 1), T2_batches, 'o-')
# plt.axhline(ucl, color='red', linestyle='--', label=f'UCL = {ucl:.2f}')
# plt.axhline(0, color='gray', linestyle=':')
# plt.xlabel('Batch'); plt.ylabel('T²')
# plt.title("Hotelling's T² control chart (batch = 10)")
# plt.legend(); plt.xticks(range(1, n_batches + 1))
# plt.savefig('t2_chart.png')
# plt.show()


"""
Consider the first two components from the data set in the previous exer-
cise. Assuming the data are normal:
    (a) Plot the data along with a 95% (elliptic) tolerance zone.
    (b) Plot the sample mean along with a 95% (elliptic) confidence zone.
    (c) Add to the previous plots (in a different color) the corresponding
        asymptotic tolerance zone and asymptotic confidence zone.
"""
X_twod = X[:, :2]
mu_twod = X_twod.mean(axis=0)
cov_twod = np.cov(X_twod.T)
n, p = X_twod.shape

vals, vecs = np.linalg.eigh(cov_twod)
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

F_crit = stats.f.ppf(0.95, p, n - p)
chi_crit = chi2.ppf(0.95, p)

# scaling factors
tol_exact  = (n - 1) * (n + 1) / (n * (n - p)) * p * F_crit
conf_exact = (n - 1) * p / (n - p) * F_crit

def ellipse_wh(scale, divide_by_n=False):
    lam = vals / n if divide_by_n else vals
    w = 2 * np.sqrt(scale * lam[0])
    h = 2 * np.sqrt(scale * lam[1])
    return w, h

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) tolerance zone
ax = axes[0]
ax.scatter(X_twod[:, 0], X_twod[:, 1], label='data')
ax.scatter(*mu_twod, c='red', label='mean')
w, h = ellipse_wh(tol_exact)
ax.add_patch(Ellipse(xy=mu_twod, width=w, height=h, angle=angle,
                     edgecolor='red', facecolor='none', lw=2, label='exact 95% tolerance'))
w, h = ellipse_wh(chi_crit)
ax.add_patch(Ellipse(xy=mu_twod, width=w, height=h, angle=angle,
                     edgecolor='purple', facecolor='none', lw=2, ls='--', label='asymptotic 95% tolerance'))
ax.set_title('(a) Tolerance zone'); ax.legend()

# (b) confidence zone
ax = axes[1]
ax.scatter(X_twod[:, 0], X_twod[:, 1], alpha=0.4, label='data')
ax.scatter(*mu_twod, c='red', label='mean')
w, h = ellipse_wh(conf_exact, divide_by_n=True)
ax.add_patch(Ellipse(xy=mu_twod, width=w, height=h, angle=angle,
                     edgecolor='red', facecolor='none', lw=2, label='exact 95% confidence'))
w, h = ellipse_wh(chi_crit, divide_by_n=True)
ax.add_patch(Ellipse(xy=mu_twod, width=w, height=h, angle=angle,
                     edgecolor='purple', facecolor='none', lw=2, ls='--', label='asymptotic 95% confidence'))
ax.set_title('(b) Confidence zone'); ax.legend()

plt.tight_layout()
plt.savefig('ex2.4_twod.png')
plt.show()