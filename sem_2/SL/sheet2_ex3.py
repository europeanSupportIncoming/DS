import numpy as np
import matplotlib.pyplot as plt
import time

def make_samples(d):
    #size: (#entries, dim)
    return np.random.normal(size=(150, d))

#1
def make_plots(d):
    if d == 1:
        samples = make_samples(d)
        plt.scatter(samples, np.zeros_like(samples), s=0.7, c='blue')
        plt.scatter(1, 0, c='red')
        plt.show()

    elif d==2:
        samples = make_samples(d)
        Circle=plt.Circle((0,0), radius=np.sqrt(d), color='red', fill=False)
        fig, ax = plt.subplots()
        ax.add_patch(Circle)
        ax.scatter(samples[:,0], samples[:,1], s=0.7, c='blue')
        plt.show()
    else:
        print(f'dim {d} not supported')

make_plots(1)
make_plots(2)


#2
def make_histo(d):
    samples = make_samples(d)
    vals = [np.linalg.norm(s) for s in samples]

    mean = np.sqrt(d)  # E||X|| ≈ sqrt(d)
    std = 1 / np.sqrt(2)

    plt.hist(vals)

    plt.axvline(mean, color='red', linestyle='--', label=f'sqrt(d) = {mean:.3f}')
    plt.axvline(mean - 2 * std, color='red', linestyle=':', label='mean pm 2 std')
    plt.axvline(mean + 2 * std, color='red', linestyle=':')

    plt.title(f'Histogram, {d} dimensions')
    plt.legend()
    plt.show()


for d in [1,2,100,1000, 100000]:
    make_histo(d)

