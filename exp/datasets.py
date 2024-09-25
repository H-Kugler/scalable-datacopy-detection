import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import os
import sys

path = os.getcwd()
sys.path.append(path)
from src.autoencoding import MNISTAutoencoder
from src.generative import *


### Visualization ###
# load the datasets
print("Load datasets")
X_hf = Halfmoons(noise=0.1).sample(1000)
X_sr = SwissRoll(noise=0.1).sample(3000)
X_gmm = GaussianMixture(n_components=3, n_features=4).sample(4000)
X_mnist = MNIST(root="./data.nosync").fit().sample(20000)

# map mnist data to 2-dim space using t-SNE
print("Map datasets to 2-dim space using t-SNE")
print("... for Swiss Roll")
tsne = TSNE(n_components=2, perplexity=15, n_jobs=4)
X_sr_tsne = tsne.fit(X_sr)
print("... for Gaussian Mixture")
tsne = TSNE(n_components=2, perplexity=20, n_jobs=4)
X_gmm_tsne = tsne.fit(X_gmm)
print("... for MNIST")
tsne = TSNE(n_components=2, perplexity=30, n_jobs=4)
X_mnist_tsne = tsne.fit(X_mnist)

# plot the datasets
print("Plotting embedded data")
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].scatter(X_hf[:, 0], X_hf[:, 1], c="b", s=5)
ax[0, 0].set_title("Halfmoons")
ax[0, 1].scatter(X_sr_tsne[:, 0], X_sr_tsne[:, 1], c="r", s=5)
ax[0, 1].set_title("Swiss Roll")
ax[1, 0].scatter(X_gmm_tsne[:, 0], X_gmm_tsne[:, 1], c="g", s=5)
ax[1, 0].set_title("Gaussian Mixture")
ax[1, 1].scatter(X_mnist_tsne[:, 0], X_mnist_tsne[:, 1], c="m", s=5)
ax[1, 1].set_title("MNIST")
for i in range(2):
    for j in range(2):
        ax[i, j].axis("off")

plt.tight_layout()
plt.savefig("./doc/figures/datasets.png")

### Regularity ###
print("Perform Regularity Analysis")

# Halfmoons
print("... for Halfmoons")
radii = np.linspace(0, 1, 100)
counts_hf = np.zeros((len(radii), len(X_hf)))
for i, x in enumerate(X_hf):
    for j, r in enumerate(radii):
        counts_hf[j, i] = np.sum(np.linalg.norm(X_hf - x, axis=1) < r)

# Swiss Roll
print("... for Swiss Roll")
radii = np.linspace(0, 1, 100)
counts_sr = np.zeros((len(radii), len(X_sr)))
for i, x in enumerate(X_sr):
    for j, r in enumerate(radii):
        counts_sr[j, i] = np.sum(np.linalg.norm(X_sr - x, axis=1) < r)

# Gaussian Mixture
print("... for Gaussian Mixture")
radii = np.linspace(0, 1, 100)
counts_gmm = np.zeros((len(radii), len(X_gmm)))
for i, x in enumerate(X_gmm):
    for j, r in enumerate(radii):
        counts_gmm[j, i] = np.sum(np.linalg.norm(X_gmm - x, axis=1) < r)

# MNIST ---- WARNING - This will take a long time to compute
print("... for MNIST")
radii = np.linspace(0, 1, 100)
subset = X_mnist[np.random.choice(len(X_mnist), 1000, replace=False)]
counts_mnist = np.zeros((len(radii), len(subset)))
for i, x in enumerate(subset):
    for j, r in enumerate(radii):
        counts_mnist[j, i] = np.sum(np.linalg.norm(X_mnist - x, axis=1) < r)
        if i % 100 == 0 and j == 0:
            print(f"Progress: {i}/{len(subset)}")

# plot mean and std of counts
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(radii, np.mean(counts_hf, axis=1), c="b")
axs[0, 0].fill_between(
    radii,
    np.mean(counts_hf, axis=1) - np.std(counts_hf, axis=1),
    np.mean(counts_hf, axis=1) + np.std(counts_hf, axis=1),
    color="b",
    alpha=0.3,
)
axs[0, 0].set_title("Regularity of Halfmoons (2D)")

axs[0, 1].plot(radii, np.mean(counts_sr, axis=1), c="r")
axs[0, 1].fill_between(
    radii,
    np.mean(counts_sr, axis=1) - np.std(counts_sr, axis=1),
    np.mean(counts_sr, axis=1) + np.std(counts_sr, axis=1),
    color="r",
    alpha=0.3,
)
axs[0, 1].set_title("Regularity of Swiss Roll (3D)")

axs[1, 0].plot(radii, np.mean(counts_gmm, axis=1), c="g")
axs[1, 0].fill_between(
    radii,
    np.mean(counts_gmm, axis=1) - np.std(counts_gmm, axis=1),
    np.mean(counts_gmm, axis=1) + np.std(counts_gmm, axis=1),
    color="g",
    alpha=0.3,
)
axs[1, 0].set_title("Regularity of Gaussian Mixture (4D)")

axs[1, 1].plot(radii, np.mean(counts_mnist, axis=1), c="m")
axs[1, 1].fill_between(
    radii,
    np.mean(counts_mnist, axis=1) - np.std(counts_mnist, axis=1),
    np.mean(counts_mnist, axis=1) + np.std(counts_mnist, axis=1),
    color="m",
    alpha=0.3,
)
axs[1, 1].set_title("Regularity of MNIST (64D)")

for ax in axs.flat:
    ax.set(xlabel="Radius", ylabel="Count")

plt.tight_layout()
plt.savefig("./doc/figures/regularity.png")
