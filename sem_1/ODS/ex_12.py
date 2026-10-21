import numpy as np
import matplotlib.pyplot as plt
import time


def f(x, k):
    return (x[1] - 3) ** 2 + k * (x[0] - 2) ** 2


def nabla_f(x, k):
    return np.array([2 * k * (x[0] - 2), 2 * (x[1] - 3)])

xax = []
yax = []
# heavy ball
for k in [10., 0.01]:
    L = 2 * np.max([k, 1])
    mu = 2 * np.min([1, k])
    alpha = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    x0 = np.array([0., 0.])

    maxiter = 1000
    eps = 1e-5
    # heavy ball also needs x_-1
    x_heavy_ball = np.eye(maxiter + 1, 2)
    x_heavy_ball[0] = x0
    x_heavy_ball[1] = x0

    i = 0
    t0 = time.time()
    while i < maxiter:
        x_heavy_ball[i + 1] = x_heavy_ball[i] - alpha * nabla_f(x_heavy_ball[i], k) + beta * (
                    x_heavy_ball[i] - x_heavy_ball[i - 1])
        i += 1
        if np.linalg.norm(nabla_f(x_heavy_ball[i], k)) < eps:
            break

    print(
        f'heavy ball method with k = {k} converged after {i} iterations with ||nabla f|| ='
        f' {np.linalg.norm(nabla_f(x_heavy_ball[i], k)):.6f} in {(time.time() - t0):.4f} seconds')
    xax.append(np.linspace(0, i, i))
    yax.append(np.log(np.array([f(x_heavy_ball[it], k) for it in range(i)] - f(x_heavy_ball[i], k))))

# nesterov optimal
for k in [10., 0.01]:

    L = 2 * np.max([k, 1])
    maxiter = 1000
    eps = 1e-5

    x0 = np.array([0., 0.])
    # set up nesterov
    tau0 = 1
    y0 = x0.copy()

    x_nesterov = np.eye(maxiter, 2) * 0
    y_nesterov = np.eye(maxiter, 2) * 0
    tau_nesterov = np.zeros(maxiter)

    x_nesterov[0] = x0
    y_nesterov[0] = y0
    tau_nesterov[0] = tau0

    i = 0
    t0 = time.time()
    while i < (maxiter - 1):
        x_nesterov[i + 1] = y_nesterov[i] - nabla_f(y_nesterov[i], k) / L
        tau_nesterov[i + 1] = (1 + np.sqrt(4 * tau_nesterov[i] ** 2 + 1)) / 2
        y_nesterov[i + 1] = x_nesterov[i] + (tau_nesterov[i] - 1) / tau_nesterov[i + 1] * (
                x_nesterov[i + 1] - x_nesterov[i])
        i += 1
        if np.linalg.norm(nabla_f(x_nesterov[i], k)) < eps:
            break

    print(
        f'nesterov method with k = {k} converged after {i} iterations with ||nabla f|| = '
        f' {np.linalg.norm(nabla_f(x_nesterov[i], k)):.6f} in {(time.time() - t0):.4f} seconds')
    xax.append(np.linspace(0, i, i))
    yax.append(np.log(np.array([f(y_nesterov[it], k) for it in range(i)] - f(y_nesterov[i], k))))

# nesterov strong
for k in [10., 0.01]:

    L = 2 * np.max([k, 1])
    maxiter = 1000
    eps = 1e-5

    x0 = np.array([0., 0.])
    # set up nesterov
    tau0 = 1
    y0 = x0.copy()

    x_nesterov_c = np.eye(maxiter, 2) * 0
    y_nesterov_c = np.eye(maxiter, 2) * 0
    tau_nesterov_c = np.zeros(maxiter)

    x_nesterov_c[0] = x0
    y_nesterov_c[0] = y0
    tau_nesterov_c[0] = tau0

    i = 0
    t0 = time.time()
    while i < (maxiter - 1):
        x_nesterov_c[i + 1] = y_nesterov_c[i] - nabla_f(y_nesterov_c[i], k) / L
        y_nesterov_c[i + 1] = x_nesterov_c[i + 1] + (1 - np.sqrt(i)) / (1 + np.sqrt(i)) * (
                x_nesterov_c[i + 1] - x_nesterov_c[i])
        i += 1
        if np.linalg.norm(nabla_f(x_nesterov_c[i], k)) < eps:
            break

    print(
        f'strongly convex nesterov method with k = {k} converged after {i} iterations with ||nabla f|| = '
        f' {np.linalg.norm(nabla_f(x_nesterov_c[i], k)):.6f} in {(time.time() - t0):.4f} seconds')
    xax.append(np.linspace(0, i, i))
    yax.append(np.log(np.array([f(y_nesterov_c[it], k) for it in range(i)] - f(y_nesterov_c[i], k))))


for k in [10., 0.01]:

    L = 2 * np.max([k, 1])
    mu = 2 * np.min([1, k])

    maxiter = 1000
    eps = 1e-5

    # set up gradient descent
    for steplength in [1 / L, 2 / (mu + L)]:

        x0 = np.array([0., 0.])
        x_grad = np.eye(maxiter, 2) * 0
        x_grad[0] = x0

        i = 0
        t0 = time.time()
        while i < (maxiter - 1):
            x0 -= steplength * nabla_f(x_grad[i], k)
            x_grad[i + 1] = x0
            i += 1
            if np.linalg.norm(nabla_f(x0, k)) < eps:
                break

        print(
            f'gradient descent method with k = {k} and step length {steplength} converged after {i} iterations with ||nabla f|| ='
            f'  {np.linalg.norm(nabla_f(x0, k)):.6f} in {(time.time() - t0):.4f} seconds')
        xax.append(np.linspace(0, i, i))
        yax.append(np.log(np.array([f(x_grad[it], k) for it in range(i)] - f(x0, k))))

plt.plot(xax[0], yax[0], label = 'heavy_ball')
plt.plot(xax[2], yax[2], label = 'nesterov')
plt.plot(xax[4], yax[4], label = 'nesterov_c')
plt.plot(xax[6], yax[6], label = 'gd')
plt.legend()
plt.title('k = 10')
plt.xlabel('iterations')
plt.ylabel('||nabla f||')
plt.show()


plt.plot(xax[1], yax[1], label = 'heavy_ball')
plt.plot(xax[3], yax[3], label = 'nesterov')
plt.plot(xax[5], yax[5], label = 'nesterov_c')
plt.plot(xax[7], yax[7], label = 'gd')
plt.legend()
plt.title('k = 0.01')
plt.xlabel('iterations')
plt.ylabel('||nabla f||')
plt.show()