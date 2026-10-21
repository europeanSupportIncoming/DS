import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Loss(theta, x_list, y_list):
    L = 0
    for x, y in zip(x_list, y_list):
        p = sigmoid(np.dot(theta, x))
        L -= (y * np.log(p) + (1 - y) * np.log(1 - p))
    return L

def nabla_L(theta, x_list, y_list):
    output = np.zeros_like(theta)
    for x, y in zip(x_list, y_list):
        output += x *( sigmoid(np.dot(theta, x)) - y)
    return output


def get_theta(max_iters = 20000, eta = 0.01, theta = np.array([0.,0.]), machine_eps = 1e-10, x_list = [], y_list = [], loss_fct = Loss):
    iters = 0
    thetas = np.eye(max_iters, len(theta))*0
    losses = np.zeros(max_iters)
    thetas[0] = theta
    losses[0] = loss_fct(theta, x_list, y_list)
    while np.abs(np.sum(nabla_L(theta, x_list, y_list))) > machine_eps and iters < (max_iters-1):
        iters += 1
        theta -= eta * nabla_L(theta, x_list, y_list)
        thetas[iters] = theta
        losses[iters] = loss_fct(theta, x_list, y_list)
    return thetas, losses, iters


x_list_ex_2= np.array([(1.,0.), (0.,1.), (-1.,0.), (0., -1.), (0.2,0.2)])
y_list_ex_2= np.array([1., 1., 0., 0., 0.])
thetas_ex2, losses_ex2, iters_ex2 = get_theta(x_list=x_list_ex_2, y_list=y_list_ex_2)
print('iterations: ', iters_ex2, 'theta final: ', thetas_ex2[iters_ex2],'final loss: ', losses_ex2[iters_ex2])


xax  =np.linspace(0, iters_ex2, iters_ex2)
y_ex2_theta = np.array([np.linalg.norm(thetas_ex2[iter]) for iter in range(iters_ex2)])
y_ex2_loss = losses_ex2[:iters_ex2].copy()
plt.plot(xax, y_ex2_theta, c='red', label='norm of theta')
plt.plot(xax, y_ex2_loss, c='blue', label ='loss')
plt.xlabel('iterations')
plt.ylabel('behaviour over iterations')
plt.legend()
plt.show()

x_list_ex3 = np.array([(2.,2.), (3.,2.), (-2.,-2.), (-3., -2.)])
y_list_ex3 = np.array([1., 1., 0., 0.])

thetas_ex3, losses_ex3, iters_ex3 = get_theta(max_iters = 100000, x_list=x_list_ex3, y_list=y_list_ex3)
print('iterations: ', iters_ex3, 'theta final: ', thetas_ex3[iters_ex3],'final loss: ', losses_ex3[iters_ex3])
xax  =np.linspace(0, iters_ex3, iters_ex3)
y_ex3_theta = np.array([np.linalg.norm(thetas_ex3[iter]) for iter in range(iters_ex3)])
y_ex3_loss = losses_ex3[:iters_ex3].copy()
plt.plot(xax, y_ex3_theta, c='red', label='norm of theta')
plt.plot(xax, y_ex3_loss, c='blue', label ='loss')
plt.xlabel('iterations')
plt.ylabel('behaviour over iterations')
plt.legend()
plt.show()


def reg_loss(theta, x_list, y_list, lamda = 1e-6):
    L = 0
    for x, y in zip(x_list, y_list):
        p = sigmoid(np.dot(theta, x))
        L -= (y * np.log(p) + (1 - y) * np.log(1 - p))
    L += lamda/2 * np.linalg.norm(theta)**2
    return L

thetas_ex4, losses_ex4, iters_ex4 = get_theta(max_iters = 100000, x_list=x_list_ex3, y_list=y_list_ex3, loss_fct=reg_loss)
print('iterations: ', iters_ex4, 'theta final: ', thetas_ex4[iters_ex4],'final loss: ', losses_ex4[iters_ex4])
xax  =np.linspace(0, iters_ex4, iters_ex4)
y_ex4_theta = np.array([np.linalg.norm(thetas_ex4[iter]) for iter in range(iters_ex4)])
y_ex4_loss = losses_ex4[:iters_ex4].copy()
plt.plot(xax, y_ex4_theta, c='red', label='norm of theta')
plt.plot(xax, y_ex4_loss, c='blue', label ='loss')
plt.xlabel('iterations')
plt.ylabel('behaviour over iterations, regularized loss')
plt.legend()
plt.show()
