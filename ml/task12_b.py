

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
import numpy as np
np.random.seed(42)
dataset,clu = make_blobs(n_samples = 1500)
model = Birch(n_clusters=4)
model.fit(dataset)
pred = model.predict(dataset)
pred
plt.scatter(dataset[:, 0], dataset[:, 1], c = pred, cmap = 'viridis')
plt.show()

