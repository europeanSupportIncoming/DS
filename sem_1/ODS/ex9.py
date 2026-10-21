import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(10)

N=100
inputs = np.random.uniform(0,1,size=N)
outputs = 10* inputs + np.random.normal(size=N)

def f(x):
    return np.sum((inputs * x - outputs) ** 2 )/(2*N)

def nabla_f(x):
    return np.sum(inputs * (inputs * x - outputs))

def stoch_gd(max_epochs, tk, iters_per_epoch = N, ins=inputs, outs=outputs, tol=1e-5):
    assert N <= len(ins), 'multi iteration problem'
    x = random.random()
    xlist = np.zeros((max_epochs, iters_per_epoch))
    in_copy = ins.copy()
    out_copy = outs.copy()
    in_out = np.stack((in_copy, out_copy), axis=1)
    for epoch in range(max_epochs):
        for iter in range(iters_per_epoch):
            random.shuffle(in_out)
            rndin = in_out[:,0]
            rndout = in_out[:,1]
            for k in range(iters_per_epoch):
                x -= tk(k) * rndin[k%N] * (rndin[k%N] * x - rndout[k%N])
            xlist[epoch, iter] = x
    return xlist

def tk_1(k):
    return 1 / np.sqrt(k + 1)

def tk_2(k):
    return 1 / np.sqrt(k + 1)

def tk_3(k, data_set_size=N):
    return 1 / np.sqrt(data_set_size + 1)

for max_epochs in [10, 1000]:
    method_one = stoch_gd(1, tk_1, N*max_epochs)
    method_two = stoch_gd(max_epochs, tk_2)
    method_three = stoch_gd(max_epochs, tk_3)


    for i, x  in enumerate(method_one[0]):
        plt.scatter(i, f(x), c='b')
    plt.title(f" method 1, {N*max_epochs} iterations one epoch, tk = 1/sqrt(k+1)")
    plt.show()

    for epoch in range(max_epochs):
        plt.scatter(epoch, f(method_two[epoch]), c='r')
    plt.title(f"method 2 epoch: {epoch}, {N} iterations per epoch, tk = 1/sqrt(k+1)")
    plt.show()

    for epoch in range(max_epochs):
        plt.scatter(epoch, f(method_three[epoch]), c='g')
    plt.title(f"method 3 epoch: {epoch}, {N} iterations per epoch, tk = 1/sqrt(N+1)")
    plt.show()

