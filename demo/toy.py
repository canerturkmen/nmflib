"""

IPython script to create and test clustering algorithms on toy data sets

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500

datasets = ["noisy_circles", "noisy_moons", "blobs", "random"]

data = [
    datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05),
    datasets.make_moons(n_samples=n_samples, noise=.05),
    datasets.make_blobs(n_samples=n_samples, random_state=8),
    np.random.rand(n_samples, 2)
]


for i, d in enumerate(data):
    pass
