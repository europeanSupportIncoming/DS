import pyreadr
import numpy as np
import matplotlib.pyplot as plt
import itertools

data = pyreadr.read_r('.././children.Rdata')

all = data['children']['exposed']
dead = data['children']['y']

xax = np.arange(len(all))
plt.figure()
plt.subplot(2,1,1)
# plt.plot(xax, all)
plt.scatter(xax, all, s=0.5, c='red')
plt.title('number of children')
plt.subplot(2,1,2)
# plt.plot(xax, dead)
plt.scatter(xax, dead, s=0.5, c='red')
plt.title('number of exposed children')
# plt.savefig('basic_EDA.png')
plt.show()

all_sliced = itertools.batched(all, 12)
dead_sliced = itertools.batched(dead, 12)
months_all = np.zeros(12)
months_dead = np.zeros(12)

for slice in all_sliced:
    months_all += np.array(slice)

for slice in dead_sliced:
    months_dead += np.array(slice)
plt.figure()
plt.subplot(1,2,1)
plt.scatter(np.arange(12), months_all, c='red')
plt.title('all kids')
plt.subplot(1,2,2)
plt.scatter(np.arange(12), months_dead, c='red')
plt.title('dead kids')
# plt.savefig('EDA_by_months.png')