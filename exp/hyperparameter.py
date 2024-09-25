import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# import Parameter grid to loop over
from sklearn.model_selection import ParameterGrid
import os
import sys

sys.path.append(os.getcwd())
from src.generative import *
from src.detection import DataCopyingDetector as DCD


##### Investigate Relationship of Hyperparameters in different Dimensions #####
X = Halfmoons(0.1).sample(2000)

n_projections = np.array([1, 10, 20, 30, 40, 50])  # , 60, 70, 80, 90, 100])
lmbdas = np.arange(1, 20, 1)
rhos = np.round(np.linspace(0.1, 0.9, 9)[::-1], 1)

pg = ParameterGrid({"rho": rhos, "lmbda": lmbdas, "n_proj": n_projections})

res = []

for params in tqdm(pg):
    l = params["lmbda"]
    r = params["rho"]
    n_proj = params["n_proj"]

    det = DCD(lmbda=l, gamma=1 / 4000)
    q = MixedModel(rho=r, r_copying=0.02, r_underfit=0.25)
    q.fit(X)

    cr = det.estimate_cr_multiple_proj(X, q, 200000, n_proj)

    res.append({"rho": r, "lmbda": l, "n_proj": n_proj, "cr": cr})

pd.DataFrame(res).to_csv("./doc/exp/hyperparameter.csv", index=False)


# ### Influence of Number of Projections ###
# X_hf = Halfmoons(0.1).sample(2000)


# detector = DCD(lmbda=10, gamma=1 / 4000)

# res_cr_q = np.zeros((len(rhos), len(n_projections)))

# for i, r in enumerate(tqdm(rhos)):
#     gen = MixtureModel(rho=r)
#     gen.fit(X_hf)
#     for j, n_proj in enumerate(n_projections):
#         res_cr_q[i, j] = detector.estimate_cr_multiple_proj(X_hf, gen, 200000, n_proj)

# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i, ax in enumerate(axs.flat):
#     ax.plot(n_projections, res_cr_q[i, :])
#     ax.axhline(rhos[i], color="r", linestyle="--", label="Theoretical Value")
#     ax.set_title(f"CR(q) for cr = {rhos[i]}")
#     ax.set_xlabel("Number of Projections")
#     ax.set_ylabel("CR(q)")
# plt.tight_layout()
# plt.savefig("./doc/figures/hyperparameter_n_proj.png")

# ### Influence of Lambda ###


# cr_q_1D = np.zeros((len(rhos), len(lmbdas)))
# cr_q_2D = np.zeros((len(rhos), len(lmbdas)))

# for i, r in enumerate(tqdm(rhos)):
#     gen = MixtureModel(rho=r)
#     gen.fit(X_hf)
#     for j, l in enumerate(lmbdas):
#         detector = DCD(lmbda=l, gamma=1 / 4000)
#         cr_q_1D[i, j] = detector.estimate_1D_cr(X_hf, gen, 200000)
#         cr_q_2D[i, j] = detector.estimate_cr(X_hf, gen, 200000)


# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# for i, ax in enumerate(axs.flat):
#     ax.plot(lmbdas, cr_q_1D[i, :], label="1D")
#     ax.plot(lmbdas, cr_q_2D[i, :], label="2D")
#     ax.axhline(rhos[i], color="r", linestyle="--", label="Theoretical Value")
#     ax.set_title(f"CR(q) for cr = {rhos[i]}")
#     ax.set_xlabel("Lambda")
#     ax.set_ylabel("CR(q)")
#     ax.legend()
# plt.tight_layout()
# plt.savefig("./doc/figures/hyperparameter_lambda.png")
