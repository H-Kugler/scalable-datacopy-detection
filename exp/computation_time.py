import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
import os
import sys

sys.path.append(os.getcwd())
from src.generative import *
from src.autoencoding import MNISTAutoencoder
from src.detection import DataCopyingDetector as DCD

##### Investigate Influence of Data Set Size and Generated Data Size #####
# Runtime: ~ 15.5 min
dataset = Halfmoons(0.1)
gen_model = MixedModel()
detector = DCD(lmbda=20, gamma=1 / 4000)


train_lengths = np.array([2000, 5000, 10000, 20000, 50000], dtype=int)
gen_lengths = np.linspace(20000, 200000, 5, dtype=int)
times_lengths = np.zeros((len(train_lengths), len(gen_lengths)))
cr_q_lengths = np.zeros((len(train_lengths), len(gen_lengths)))

for i, tl in enumerate(tqdm(train_lengths)):
    X = dataset.sample(tl)
    gen_model.fit(X)
    for j, gl in enumerate(gen_lengths):
        time_start = time()
        cr_q_lengths[i, j] = detector.estimate_cr(S=X, q=gen_model, m=gl)
        time_end = time()
        times_lengths[i, j] = time_end - time_start

# create 3D plot of results
fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(gen_lengths, train_lengths, indexing="ij")
for i, ax in enumerate(axs):
    if i == 0:
        Z = times_lengths
        ax.set_title("Time")
    else:
        Z = np.abs(cr_q_lengths - 0.2)
        ax.set_title("CR(q)")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Generated data size")
    ax.set_ylabel("Training data size")
    ax.set_zlabel("Value")

plt.tight_layout()
plt.savefig("./doc/figures/comp_setsize.png")


##### Investigate Influence of Dimensionality #####
X = MNIST(root="./data/").sample(20000)
enc = MNISTAutoencoder()
enc.load_state_dict(
    torch.load(
        "./data/trained_weights/trained_autoencoder_weights.pth",
        map_location=torch.device("cpu"),
    )
)
enc.eval()
X = enc.encode(X).detach().numpy()

detector = DCD(lmbda=20, gamma=1 / 4000)
gen_model = WorstDataCopier()

dims = np.array([1, 2, 4, 8, 16, 32, 64])
times_dims = np.zeros(len(dims))
cr_q_dims = np.zeros(len(dims))

for i, d in enumerate(tqdm(dims)):
    proj = GaussianRandomProjection(n_components=d)
    X_proj = proj.fit_transform(X)
    gen_model.fit(X_proj)
    time_start = time()
    cr_q_dims[i] = detector.estimate_cr(S=X_proj, q=gen_model, m=20000)
    time_end = time()
    times_dims[i] = time_end - time_start

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(dims, times_dims, "o-")
axs[0].set_xlabel("Dimensions")
axs[0].set_xticks(dims)
axs[0].set_ylabel("Time in second")
axs[0].set_title("Time to estimate CR")
axs[1].plot(dims, cr_q_dims, "o-")
axs[1].set_xlabel("Dimensions")
axs[1].set_ylabel("CR(q)")
axs[1].set_title("CR(q) vs. Dimensions")

plt.tight_layout()
plt.savefig("./doc/figures/comp_dims.png")
