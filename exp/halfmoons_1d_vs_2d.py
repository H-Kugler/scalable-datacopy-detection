import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from src.generative import MixedModel, KDE, Mixture, Halfmoons
from src.detection import DataCopyingDetector as DCD

# Comparing 1D and 2D results on Halfmoons
print("Comparing 2D with 1D results on Halfmoons")
p = Halfmoons(noise=0.1)
X_hf = p.sample(2000)  # dataset
det = DCD(lmbda=20, gamma=1 / 4000)  # data copying detector
N_gen = 1000  # number of generated samples
N_trials = 100  # number of trials

# mixed model, i.e. q = rho * q_copying + (1 - rho) * q_underfit
print("Mixed Model: q = rho * q_copying + (1 - rho) * q_underfit")
rhos = np.linspace(0.1, 0.9, 9)
crqs_2d_mm = np.zeros((len(rhos), N_trials))
crqs_1d_mm = np.zeros((len(rhos), N_trials))
for i, r in enumerate(tqdm(rhos)):
    for j in range(N_trials):
        mm = MixedModel(rho=r).fit(X_hf)
        crqs_2d_mm[i, j] = det.estimate_cr(S=X_hf, q=mm, m=N_gen)
        crqs_1d_mm[i, j] = det.estimate_1D_cr(S=X_hf, q=mm, m=N_gen)


# mixture of models, i.e. q = rho * q_copying + (1 - rho) * p
print("Mixture of Models: q = rho * q_copying + (1 - rho) * p")
crqs_2d_mixture = np.zeros((len(rhos), N_trials))
crqs_1d_mixture = np.zeros((len(rhos), N_trials))
for i, r in enumerate(tqdm(rhos)):
    for j in range(N_trials):
        mixture = Mixture(rho=r, q1=MixedModel(rho=1), q2=p).fit(X_hf)
        crqs_2d_mixture[i, j] = det.estimate_cr(S=X_hf, q=mixture, m=N_gen)
        crqs_1d_mixture[i, j] = det.estimate_1D_cr(S=X_hf, q=mixture, m=N_gen)

# kde
print("KDE")
sigmas = np.logspace(-6, 0, 20)
crqs_2d_kde = np.zeros((len(sigmas), N_trials))
crqs_1d_kde = np.zeros((len(sigmas), N_trials))
for i, s in enumerate(tqdm(sigmas)):
    for j in range(N_trials):
        kde = KDE(bandwidth=s).fit(X_hf)
        crqs_2d_kde[i, j] = det.estimate_cr(S=X_hf, q=kde, m=N_gen)
        crqs_1d_kde[i, j] = det.estimate_1D_cr(S=X_hf, q=kde, m=N_gen)

# plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("1D vs 2D on Halfmoons")

axs[0].errorbar(
    rhos,
    np.mean(crqs_1d_mm, axis=1),
    yerr=np.std(crqs_1d_mm, axis=1),
    fmt="o",
    label="1D",
)
axs[0].errorbar(
    rhos,
    np.mean(crqs_2d_mm, axis=1),
    yerr=np.std(crqs_2d_mm, axis=1),
    fmt="o",
    label="2D",
)
axs[0].plot([0, 1], [0, 1], "--", color="black", label="Ground Truth")
axs[0].set_xlabel("$\\rho$")
axs[0].set_ylabel("$cr_q$")
axs[0].set_title("$q = \\rho * q_{copying} + (1 - \\rho) * q_{underfit}$")

axs[1].errorbar(
    rhos,
    np.mean(crqs_1d_mixture, axis=1),
    yerr=np.std(crqs_1d_mixture, axis=1),
    fmt="o",
    label="1D",
)
axs[1].errorbar(
    rhos,
    np.mean(crqs_2d_mixture, axis=1),
    yerr=np.std(crqs_2d_mixture, axis=1),
    fmt="o",
    label="2D",
)
axs[1].plot([0, 1], [0, 1], "--", color="black", label="Ground Truth")
axs[1].set_xlabel("$\\rho$")
axs[1].set_ylabel("$cr_q$")
axs[1].set_title("$q = \\rho * q_{copying} + (1 - \\rho) * p$")

axs[2].errorbar(
    sigmas,
    np.mean(crqs_1d_kde, axis=1),
    yerr=np.std(crqs_1d_kde, axis=1),
    fmt="o",
    label="1D",
)
axs[2].errorbar(
    sigmas,
    np.mean(crqs_2d_kde, axis=1),
    yerr=np.std(crqs_2d_kde, axis=1),
    fmt="o",
    label="2D",
)
axs[2].set_xscale("log")
axs[2].set_xlabel(r"$\sigma$")
axs[2].set_ylabel(r"$cr_q$")
axs[2].set_title(r"$q = 1/n * \sum_{i=1}^n \mathcal{N}(x_i | x, \sigma)$")

plt.legend()
plt.tight_layout()
plt.savefig("./doc/figures/1d_vs_2d_halfmoons.png")

# store results
np.savez(
    "./doc/exp/1d_vs_2d_halfmoons.npz",
    crqs_1d_kde=crqs_1d_kde,
    crqs_2d_kde=crqs_2d_kde,
    crqs_1d_mm=crqs_1d_mm,
    crqs_2d_mm=crqs_2d_mm,
    crqs_1d_mixture=crqs_1d_mixture,
    crqs_2d_mixture=crqs_2d_mixture,
    sigmas=sigmas,
    rhos=rhos,
)
