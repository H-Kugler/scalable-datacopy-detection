import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm
import torch
import os
import sys

sys.path.append(os.getcwd())
from src.autoencoding import MNISTAutoencoder
from src.generative import *
from src.detection import *
from src.plotting import plot_C_T_results


##### Replicating Meehan et al. (2020) #####
print("Replicating Meehan et al. (2020)")
num_sigmas = 75  # number of sigmas to test -> sigma controlls the degree of data copying in a KDE
num_trials = 5  # number of trials to average over

sigmas = np.logspace(start=-3, stop=1, base=10, num=num_sigmas)
params = ParameterGrid({"bandwidth": sigmas})


# Halfmoons
print("... on Halfmoons")
dataset = Halfmoons(0.2)

l = 2000  # number of training samples
m = 1000  # number of generated samples
n = 1000  # number of test samples
k = 5  # number of regions

X_train = dataset.sample(l)
X_test = dataset.sample(n)

det = ThreeSampleDetector(k=k)
C_Ts_hm = det.run_params(
    Q=KernelDensity,
    params=params,
    X_train=X_train,
    X_test=X_test,
    num_trials=num_trials,
)

# MNIST
# Runtime: ~  2 min
print("... on MNIST")
dataset = MNIST(root="./data.nosync").fit()
X_train = dataset.sample()
X_test = dataset.sample(n_samples=10000, S="test")

l = len(X_train)  # number of training samples
n = len(X_test)  # number of test samples
m = len(X_test)  # number of generated samples
k = 50  # number of regions

assert l == 50000 and n == 10000 and m == 10000

det = ThreeSampleDetector(k=k)
C_Ts_mnist = det.run_params(
    Q=KernelDensity,
    params=params,
    X_train=X_train,
    X_test=X_test,
    num_trials=num_trials,
)

# plotting results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_C_T_results(sigmas, C_Ts_hm, axs[0], "Halfmoons")
plot_C_T_results(sigmas, C_Ts_mnist, axs[1], "MNIST without Random Projections")
plt.tight_layout()
plt.savefig("./doc/figures/replication_meehan.png")

# Use Gaussian Random Projections to reduce dimensionality
# Runtime: ~  10 min
print("Using Gaussian Random Projections to reduce dimensionality")
k = 50  # number of regions
num_trials = 10  # number of trials to average over
n_components = np.linspace(2, 10, 5, dtype=int)
C_Ts_mnist_grp = np.zeros((len(params), num_trials, len(n_components)))

for i, n_comp in enumerate(n_components):
    print(f"n_components: {n_comp}")

    # partition the data into k regions
    kmeans = KMeans(n_clusters=k).fit(X_train)
    T_cells = kmeans.labels_
    Pn_cells = kmeans.predict(X_test)

    for j, param in enumerate(tqdm(params)):
        Q = KDE(**param).fit(X_train)

        for t in range(num_trials):
            Qm = Q.sample(m).astype("float32")  # generate samples
            Qm_cells = kmeans.predict(Qm)  # partition the generated samples
            grp = GaussianRandomProjection(n_components=n_comp)
            grp.fit(np.r_[X_train, X_test, Qm])
            X_train_rp = grp.transform(X_train)
            X_test_rp = grp.transform(X_test)
            Qm_rp = grp.transform(Qm)
            C_Ts_mnist_grp[j, t, i] = ThreeSampleDetector._C_T(
                X_test_rp,
                Pn_cells=Pn_cells,
                Qm=Qm_rp,
                Qm_cells=Qm_cells,
                T=X_train_rp,
                T_cells=T_cells,
                tau=20 / m,
            )

# plotting results
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# set figure title
fig.suptitle("MNIST with Gaussian Random Projections")
plot_C_T_results(sigmas, C_Ts_mnist, axs[0, 0], "n = 64")
for i, n_comp in enumerate(n_components):
    plot_C_T_results(
        sigmas, C_Ts_mnist_grp[:, :, i], axs.flatten()[i + 1], f"n = {n_comp}"
    )

plt.tight_layout()
plt.savefig("./doc/figures/meehan_with_grp.png")


##### Replicating Bhattacharjee et al. (2023) #####
print("Replicating Bhattacharjee et al. (2023)")
X = Halfmoons(0.1).sample(2000)
det = DataCopyingDetector(lmbda=20, gamma=1 / 4000)

# Mixture Model
print("... using Mixture Model")
rhos = np.linspace(0, 1, 11)
crqs_mix = np.zeros(len(rhos))
for i, rho in enumerate(tqdm(rhos)):
    q = MixedModel(rho=rho, r_copying=0.02, r_underfit=0.25)
    q.fit(X)
    crqs_mix[i] = det.estimate_cr(S=X, q=q, m=200000)

# KDE
print("... using KDE")
sigmas = np.logspace(-3, 0, 10)
crqs_kde = []
for i, sigma in enumerate(tqdm(sigmas)):
    kde = KDE(bandwidth=sigma)
    kde.fit(X)
    crqs_kde.append(det.estimate_cr(S=X, q=kde, m=200000))

# plotting results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(rhos, crqs_mix, "o-", label="Mixture Model")
axs[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="Ground Truth")
axs[0].set_xlabel(r"$\rho$")
axs[0].set_ylabel("CR_q")
axs[0].set_title("Mixture Model")
axs[0].legend()

axs[1].plot(sigmas, crqs_kde, "o-", label="KDE")
axs[1].set_xlabel(r"$\sigma$")
axs[1].set_xscale("log")
axs[1].set_ylabel("CR_q")
axs[1].set_title("KDE")
axs[1].legend()

plt.tight_layout()
plt.savefig("./doc/figures/replication_bhatt.png")
