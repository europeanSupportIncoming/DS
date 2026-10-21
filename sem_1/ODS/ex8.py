import numpy as np
import time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def normed_loss(theta, X, y, lamda):
    z = X @ theta
    return np.sum(np.log(1 + np.exp(z)) - y * z) + lamda / 2 * np.linalg.norm(theta)**2

def nabla_L(theta, X, y, lamda):
    z = X @ theta
    p = sigmoid(z)
    return X.T @ (p - y) + lamda * theta

def direct_max_eigval(X):
    return np.max(np.linalg.eigvals(X.T @ X))

def power_iteration(A, num_iterations=100, tol=1e-6):
    x = np.random.rand(A.shape[1])
    x /= np.linalg.norm(x)
    lamda_prev = None
    for i in range(num_iterations):
        y = A @ x
        x = y / np.linalg.norm(y)
        lamda = (x.T @ (A @ x)).item()
        if lamda_prev is not None and abs(lamda - lamda_prev) < tol:
            break
        lamda_prev = lamda
    return lamda

def strong_convexity_param(X, lamda):
    eigvals = np.linalg.eigvals(X.T @ X)
    return np.min(eigvals) + lamda

def line_search(theta, d, X, y, lamda):
    def tmpfct(t):
        return normed_loss(theta + t * d, X, y, lamda)
    a, b = 0.0, 2.0
    for _ in range(100):
        t1 = a + (b - a) * 0.382
        t2 = a + (b - a) * 0.618
        if tmpfct(t1) < tmpfct(t2):
            b = t2
        else:
            a = t1
        if b - a < 1e-8:
            break
    return (a + b) / 2

def backtrack(theta, d, X, y, lamda, alpha, beta):
    t = 1.0
    tmp_f = normed_loss(theta, X, y, lamda)
    nabla = nabla_L(theta, X, y, lamda)
    slope = nabla @ d
    theta_new = theta + t * d
    while normed_loss(theta_new, X, y, lamda) > tmp_f + alpha * t * slope:
        t *= beta
        theta_new = theta + t * d
    return t

def gd_fixed_steps_info(theta, X, y, lamda, step_size, max_iters, tol):
    iters = 0
    n, d = X.shape
    theta_curr = theta.copy()
    grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
    times = []
    while grad_norm > tol and iters < max_iters:
        start = time.time()
        grad = nabla_L(theta_curr, X, y, lamda)
        theta_curr = theta_curr - step_size * grad
        grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
        times.append(time.time() - start)
        iters += 1
    return iters, np.mean(times) if times else 0, np.sum(times)

def gd_exact_line_search_info(theta, X, y, lamda, max_iters, tol):
    iters = 0
    n, d = X.shape
    theta_curr = theta.copy()
    grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
    times = []
    while grad_norm > tol and iters < max_iters:
        start = time.time()
        grad = nabla_L(theta_curr, X, y, lamda)
        alpha = line_search(theta_curr, -grad, X, y, lamda)
        theta_curr = theta_curr - alpha * grad
        grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
        times.append(time.time() - start)
        iters += 1
    return iters, np.mean(times) if times else 0, np.sum(times)

def gd_backtracking_info(theta, X, y, lamda, max_iters, tol, alpha, beta):
    iters = 0
    n, d = X.shape
    theta_curr = theta.copy()
    grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
    times = []
    while grad_norm > tol and iters < max_iters:
        start = time.time()
        grad = nabla_L(theta_curr, X, y, lamda)
        t = backtrack(theta_curr, -grad, X, y, lamda, alpha, beta)
        theta_curr = theta_curr - t * grad
        grad_norm = np.linalg.norm(nabla_L(theta_curr, X, y, lamda))
        times.append(time.time() - start)
        iters += 1
    return iters, np.mean(times) if times else 0, np.sum(times)

if __name__ == "__main__":
    np.random.seed(42)
    n, d = 100, 10
    X = np.random.randn(n, d)
    true_theta = np.random.randn(d)
    y = (sigmoid(X @ true_theta) > 0.5).astype(float)
    lamda = 0.01
    theta0 = np.zeros(d)
    max_iters = 1000
    tol = 1e-6

    L_direct = direct_max_eigval(X)
    L_power = power_iteration(X.T @ X)
    mu = strong_convexity_param(X, lamda)

    step_sizes = [1/L_direct, 1/L_power, 2/(mu + L_direct), 2/(mu + L_power)]
    alphas = [0.1, 0.3, 0.5]
    betas = [0.3, 0.5, 0.8]

    print("\n" + "="*100)
    print(f"{'Method':<40} {'Info needed':<20} {'Time/step (s)':<15} {'Iterations':<12} {'Total time (s)':<15}")
    print("="*100)

    iters_fixed1, time_per_fixed1, total_fixed1 = gd_fixed_steps_info(theta0, X, y, lamda, step_sizes[0], max_iters, tol)
    print(f"{'Fixed (1/L_direct)':<40} {'L':<20} {time_per_fixed1:<15.6f} {iters_fixed1:<12} {total_fixed1:<15.6f}")

    iters_fixed2, time_per_fixed2, total_fixed2 = gd_fixed_steps_info(theta0, X, y, lamda, step_sizes[1], max_iters, tol)
    print(f"{'Fixed (1/L_power)':<40} {'L':<20} {time_per_fixed2:<15.6f} {iters_fixed2:<12} {total_fixed2:<15.6f}")

    iters_fixed3, time_per_fixed3, total_fixed3 = gd_fixed_steps_info(theta0, X, y, lamda, step_sizes[2], max_iters, tol)
    print(f"{'Fixed (2/(mu+L_direct))':<40} {'L, mu':<20} {time_per_fixed3:<15.6f} {iters_fixed3:<12} {total_fixed3:<15.6f}")

    iters_fixed4, time_per_fixed4, total_fixed4 = gd_fixed_steps_info(theta0, X, y, lamda, step_sizes[3], max_iters, tol)
    print(f"{'Fixed (2/(mu+L_power))':<40} {'L, mu':<20} {time_per_fixed4:<15.6f} {iters_fixed4:<12} {total_fixed4:<15.6f}")

    iters_exact, time_per_exact, total_exact = gd_exact_line_search_info(theta0, X, y, lamda, max_iters, tol)
    print(f"{'Exact Line Search':<40} {'None':<20} {time_per_exact:<15.6f} {iters_exact:<12} {total_exact:<15.6f}")

    for alpha in alphas:
        for beta in betas:
            iters_bt, time_per_bt, total_bt = gd_backtracking_info(theta0, X, y, lamda, max_iters, tol, alpha, beta)
            print(f"{'Backtracking (a='+str(alpha)+', b='+str(beta)+')':<40} {'None':<20} {time_per_bt:<15.6f} {iters_bt:<12} {total_bt:<15.6f}")

    print("="*100)