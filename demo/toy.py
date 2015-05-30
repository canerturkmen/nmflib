"""

IPython script to create and test clustering algorithms on toy data sets

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from nmflib.cluster import NMFClustering

np.random.seed(0)

n_samples = 1500

dsets = ["circles", "moons", "blobs", "random"]

data = [
    datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0],
    datasets.make_moons(n_samples=n_samples, noise=.05)[0],
    datasets.make_blobs(n_samples=n_samples, random_state=8)[0],
    np.random.rand(n_samples, 2)
]




# For each, run a K-means clustering with k=2 and plot

for i, d in enumerate(data):

    d = d + np.abs(d.min())

    y_km = cluster.KMeans(2).fit_predict(d)
    y_gmm = mixture.GMM(2).fit_predict(d)
    y_spec = cluster.SpectralClustering(2).fit_predict(d)


km = cluster.KMeans(2)
# nmf = NMFClustering(2, "spectral", {"affinity": "nn", "nn":30}, 300, 1e-10)
nmf = NMFClustering(2, "cluster", {}, 300, 1e-10)
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

f, plots = plt.subplots(2,2)
for i, d0 in enumerate(data):
    # y = km.fit_predict(d)

    #project to nonnegative quadrant
    # d = .5 * (np.abs(d) + d)
    np.random.shuffle(d0)
    d = d0[:500,:]

    d = d + np.abs(d.min())

    y, result = nmf.fit_predict(d + 15)

    plots[i%2, i/2].set_title(dsets[i])
    plots[i%2, i/2].scatter(d[:,0], d[:,1], marker=".", color=colors[y].tolist())

plt.show()
