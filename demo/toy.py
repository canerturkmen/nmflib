"""

IPython script to create and test clustering algorithms on toy data sets

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500

dsets = ["circles", "moons", "blobs", "random"]

data = [
    datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0],
    datasets.make_moons(n_samples=n_samples, noise=.05)[0],
    datasets.make_blobs(n_samples=n_samples, random_state=8)[0],
    np.random.rand(n_samples, 2)
]


f, plots = plt.subplots(2,2)
# First provide an overview of the data
for i, d in enumerate(data):

    plots[i%2, i/2].set_title(dsets[i])
    plots[i%2, i/2].scatter(d[:,0], d[:,1], s=100)

plt.show()

# For each, run a K-means clustering with k=2 and plot

km = cluster.KMeans(2)
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

f, plots = plt.subplots(2,2)
for i, d in enumerate(data):
    y = km.fit_predict(d)

    plots[i%2, i/2].set_title(dsets[i])
    plots[i%2, i/2].scatter(d[:,0], d[:,1], s=100, color=colors[y].tolist())

plt.show()
