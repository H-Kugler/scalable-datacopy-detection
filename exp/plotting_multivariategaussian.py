import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load data
df_true_gen = pd.read_csv(
    "./doc/exp/exp_GaussianMixture_GaussianMixture(n_components=1, n_features=4).csv"
)
df_mixed = pd.read_csv(
    "./doc/exp/exp_MixedModel_GaussianMixture(n_components=1, n_features=4).csv"
)

######## Visualization of C_T statistics ########
fig, axs = plt.subplots()

sns.lineplot(
    x="rho",
    y="C_T",
    hue="num_regions",
    data=df_mixed,
    ax=axs,
    marker="^",
    errorbar="sd",
)
axs.set_xlabel("$\\rho$")
axs.set_ylabel("$C_T$")
axs.axhline(y=-3, color="black", linestyle="--", label="Significance level")
axs.set_title("$C_T$ for $q = \\rho * q_{copying} + (1 - \\rho) * q_{underfit}$")

plt.tight_layout()
plt.savefig("./doc/figures/C_T_MixedModel_GaussianMixture.png")
