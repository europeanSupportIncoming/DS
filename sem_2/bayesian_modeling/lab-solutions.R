# ===================
# Midge wing exercise
# ===================

library(ks)

# Load the data
x = c(1.64,1.7,1.72,1.82,1.82,1.9,2.08)
n = length(x)
sum_x = sum(x)

# NIG-prior parameters. Vary these for sensitivity analysis.
m_0 = 1.9
M_0 = 1
nu_0 = 2
sigma_0_sq = 0.01

# Conjugate posterior parameters
M_n = M_0 / (M_0 * n + 1)
m_n = M_n * (m_0 / M_0 + sum_x)
nu_n = nu_0 + n
sigma_n_sq = (nu_0 * sigma_0_sq + sum((x - sum_x / n)^2) + (sum_x / n - m_0)^2 / (M_0 + 1 / n))

# Posterior means as point estimates
post_mean_mu = M_n * (m_0 / M_0 + sum_x)
post_mean_sigma_sq = sigma_n_sq * nu_n / (nu_n - 2)

# Samples for estimating credible set
n_samples = 100000
samples = matrix(rep(NA, 2 * n_samples), nrow = n_samples)
samples[,2]= 1 / rgamma(n_samples, nu_n / 2, sigma_n_sq * nu_n / 2) # Posterior sigma_sq-draws
samples[,1] = rnorm(n_samples, m_n, sqrt(M_n * samples[,2])) # Posterior mu-draws

post_kde = kde(samples)
credible_set = contourLines(post_kde$eval.points[[1]], post_kde$eval.points[[2]], post_kde$estimate, levels = post_kde$cont["5%"])

plot(samples, col="grey", xlab="mu", ylab="sigma_squared", cex.lab=1.3, cex.axis=1.3)
points(post_mean_mu, post_mean_sigma_sq, pch=19, col="red")
lines(credible_set[[1]]$x, credible_set[[1]]$y, col="blue")
legend("topright", legend = c("Posterior samples", "Posterior mean", "95% credible set"), col=c("grey", "red", "blue"), pch=c(1,19,NA), lty=c(NA,NA,1),cex=1.3)

# ====================
# Boston rent exercise
# ====================

library(MASS)
library(mnorm)

# Fit (frequentist) linear model
lm.boston = lm(medv ~ lstat + age, MASS::Boston)
summary(lm.boston)

# Prior parameters. Vary for sensitivity analysis.
d = length(coef(lm.boston))
b_0 = rep(0, d)
B_0 = diag(10, d)
s_0 = 2
S_0 = 1
B_0_inv = solve(B_0) # store to avoid computing multiple inverses later

# Compute posterior parameters
X = model.matrix(lm.boston)
n = nrow(X)
y = MASS::Boston$medv
B_n = solve(crossprod(X) + B_0_inv)
b_n = B_n %*% (t(X) %*% y + B_0_inv %*% b_0)
s_n = s_0 + n / 2
S_n = S_0 + (t(y) %*% y + t(b_0) %*% B_0_inv %*% b_0 - t(b_n) %*% (crossprod(X) + B_0_inv) %*% b_n) / 2

n_samples = 10000
samples = matrix(rep(NA, (d + 1) * n_samples), nrow = n_samples)
samples[,d + 1] = 1 / rgamma(n_samples, s_n, S_n) # Posterior sigma_sq-draws
for (i in 1:n_samples) {
  samples[i, 1:d] = rmnorm(1, b_n, samples[i, d + 1] * B_n) # Posterior beta-draws from a multivariate normal distribution
}

# We compute credible sets for 2d marginals for ease of visualisation.
d1 = 1 # d1 and d2 are the two dimensions to be plotted
d2 = 4
post_kde = kde(samples[,c(d1, d2)])
credible_set = contourLines(post_kde$eval.points[[1]], post_kde$eval.points[[2]], post_kde$estimate, levels = post_kde$cont["5%"])

posterior_mean = c(b_n, S_n / (s_n - 1))
labels = c("beta_0", "beta_1", "beta_2", "sigma^2")
plot(samples[,c(d1, d2)], col="grey", xlab=labels[d1], ylab=labels[d2], cex.lab=1.3, cex.axis=1.3) # Remember to change the axis labels for other parameter pairs
points(posterior_mean[d1], posterior_mean[d2], pch=19, col="red")
lines(credible_set[[1]]$x, credible_set[[1]]$y, col="blue")
legend("topright", legend = c("Posterior samples", "Posterior mean", "95% credible set"), col=c("grey", "red", "blue"), pch=c(1,19,NA), lty=c(NA,NA,1),cex=1.3)
