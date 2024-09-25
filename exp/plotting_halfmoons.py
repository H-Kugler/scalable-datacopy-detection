import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df_true_gen = pd.read_csv("./doc/exp/exp_Halfmoons_Halfmoons(noise=0.1).csv")
df_mixed = pd.read_csv("./doc/exp/exp_MixedModel_Halfmoons(noise=0.1).csv")
df_mixture = pd.read_csv("./doc/exp/exp_Mixture_Halfmoons(noise=0.1).csv")

######## Visualization of C_T statistics ########

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, data in enumerate([df_mixed, df_mixture]):
    sns.lineplot(
        x="rho",
        y="C_T",
        hue="num_regions",
        data=data,
        ax=axs[i],
        marker="^",
        errorbar="sd",
    )
    axs[i].set_xlabel("$\\rho$")
    axs[i].set_ylabel("$C_T$")
    axs[i].axhline(y=-3, color="black", linestyle="--", label="Significance level")
    axs[i].set_title(
        "$C_T$ for $q = \\rho * q_{copying} + (1 - \\rho) * q_{underfit}$"
        if i == 0
        else "$C_T$ for $q = \\rho * q_{copying} + (1 - \\rho) * p$"
    )

y_min = min([ax.get_ylim()[0] for ax in axs])
y_max = max([ax.get_ylim()[1] for ax in axs])
for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("./doc/figures/C_T_rho.png")

# Zu plot
fig, ax = plt.subplots()
sns.lineplot(
    x="rho",
    y="Zu",
    hue="num_regions",
    data=df_mixed,
    ax=ax,
    marker="^",
    errorbar="sd",
)
ax.set_xlabel("$\\rho$")
ax.set_ylabel("$Zu_{min}$")
ax.axhline(y=-3, color="black", linestyle="--", label="Significance level")
ax.set_title("$Zu_{min}$ for $q = \\rho * q_{copying} + (1 - \\rho) * q_{underfit}$")

plt.tight_layout()
plt.savefig("./doc/figures/Zu_rho.png")

######## Visualization of copying rates ########
# q = p
cr_true_gen = (
    df_true_gen.groupby(["noise", "trial", "lmbda", "k", "gamma"]).mean().reset_index()
)
# q = rho * q_copying + (1 - rho) * q_underfit
cr_mixed = (
    df_mixed.groupby(["trial", "rho", "lmbda", "k", "gamma"]).mean().reset_index()
)
# q = rho * q_copying + (1 - rho) * p
cr_mixture = (
    df_mixture.groupby(["trial", "rho", "lmbda", "k", "gamma"]).mean().reset_index()
)

# test statistics
test_stat_mixture = pd.read_csv("./doc/exp/test_statistics_mixture.csv")
test_stat_mixed = pd.read_csv("./doc/exp/test_statistics_mixed_model.csv")

rhos = np.unique(df_mixed.rho)
lmbdas = np.unique(df_mixed.lmbda)
ks = np.unique(df_mixed.k)


######## Absolute copying rates ########
# mixed model
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, rho in enumerate(rhos):
    sns.violinplot(
        x="lmbda",
        y="cr_q",
        hue="k",
        data=cr_mixed[np.isclose(cr_mixed["rho"], rho)].reset_index(),
        linewidth=0.5,
        density_norm="width",
        ax=axs[i // 3, i % 3],
    )
    axs[i // 3, i % 3].set_title(f"rho = {round(rho, 1)}")
    axs[i // 3, i % 3].set_xlabel(r"$\lambda$")
    axs[i // 3, i % 3].set_ylabel("cr")

for ax in axs.flat:
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("doc/figures/cr_mixed_model.png")

# mixture model
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i, rho in enumerate(rhos):
    sns.violinplot(
        x="lmbda",
        y="cr_q",
        hue="k",
        data=cr_mixture[np.isclose(cr_mixture["rho"], rho)].reset_index(),
        linewidth=0.5,
        density_norm="width",
        ax=axs[i // 3, i % 3],
    )
    axs[i // 3, i % 3].set_title(f"rho = {round(rho, 1)}")
    axs[i // 3, i % 3].set_xlabel(r"$\lambda$")
    axs[i // 3, i % 3].set_ylabel("cr")

for ax in axs.flat:
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("doc/figures/cr_mixture_model.png")

######## Differences of copying rates compared to true distribution's copying rates ########

cr_diff_mixed = cr_mixed.merge(
    cr_true_gen.groupby(["lmbda", "k"]).mean().reset_index(),
    on=["lmbda", "k"],
    suffixes=("_mixed", "_true"),
)
cr_diff_mixed["cr_q_mixed"] = cr_diff_mixed["cr_q_mixed"] - cr_diff_mixed["cr_q_true"]
cr_diff_mixture = cr_mixture.merge(
    cr_true_gen.groupby(["lmbda", "k"]).mean().reset_index(),
    on=["lmbda", "k"],
    suffixes=("_mixture", "_true"),
)
cr_diff_mixture["cr_q_mixture"] = (
    cr_diff_mixture["cr_q_mixture"] - cr_diff_mixture["cr_q_true"]
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(
    x="rho",
    y="cr_q_mixed",
    hue="k",
    data=cr_diff_mixed,
    ax=axs[0],
    marker="^",
    errorbar="sd",
)
axs[0].set_xlabel("$\\rho$")
axs[0].set_ylabel("Difference in $cr_q$")
axs[0].set_title("Mixed model")

sns.lineplot(
    x="rho",
    y="cr_q_mixture",
    hue="k",
    data=cr_diff_mixture,
    ax=axs[1],
    marker="^",
    errorbar="sd",
)
axs[1].set_xlabel("$\\rho$")
axs[1].set_title("Mixture model")

y_min = min([ax.get_ylim()[0] for ax in axs])
y_max = max([ax.get_ylim()[1] for ax in axs])
for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("./doc/figures/cr_diff_summary.png")

# # mixed model
# # visualize for every rho: 28 subplots for each rho value
# tuples = list(itertools.product(lmbdas, ks))

# for c in rhos:

#     fig, axs = plt.subplots(len(lmbdas), len(ks), figsize=(20, 20))

#     for i, (l, k) in enumerate(tuples):
#         cr_true = cr_true_gen[(cr_true_gen.k == k) & (cr_true_gen.lmbda == l)].cr_q
#         crs = cr_mixed[
#             (cr_mixed.k == k) & (cr_mixed.lmbda == l) & (cr_mixed.rho == c)
#         ].cr_q

#         sns.histplot(cr_true, ax=axs.flatten()[i], label="true", kde=True)
#         sns.histplot(crs, ax=axs.flatten()[i], label="mixed", kde=True)
#         axs[i // len(ks), i % len(ks)].legend()

#         # add information whether t-test is significant
#         if (
#             test_stat_mixed[
#                 (test_stat_mixed.k == k)
#                 & (test_stat_mixed.lmbda == l)
#                 & (test_stat_mixed.rho == c)
#             ].t_p.values[0]
#             < 0.05
#         ):
#             axs[i // len(ks), i % len(ks)].set_title(
#                 f"k={k}, lmbda={l}, t-test significant"
#             )
#         else:
#             axs[i // len(ks), i % len(ks)].set_title(
#                 f"k={k}, lmbda={l}, t-test not significant"
#             )

#     plt.tight_layout()
#     plt.savefig(f"./doc/figures/density_comp_mixed_model_rho={round(c, 1)}.png")

# # mixture of models
# # visualize for every rho: 28 subplots for each rho value
# tuples = list(itertools.product(lmbdas, ks))

# for c in rhos:

#     fig, axs = plt.subplots(len(lmbdas), len(ks), figsize=(20, 20))

#     for i, (l, k) in enumerate(tuples):
#         cr_true = cr_true_gen[(cr_true_gen.k == k) & (cr_true_gen.lmbda == l)].cr_q
#         crs = cr_mixture[
#             (cr_mixture.k == k)
#             & (cr_mixture.lmbda == l)
#             & (np.isclose(cr_mixture.rho, c))
#         ].cr_q

#         sns.histplot(cr_true, ax=axs.flatten()[i], label="true", kde=True)
#         sns.histplot(crs, ax=axs.flatten()[i], label="mixture", kde=True)
#         axs[i // len(ks), i % len(ks)].legend()

#         # add information whether t-test is significant
#         if (
#             test_stat_mixture[
#                 (test_stat_mixture.k == k)
#                 & (test_stat_mixture.lmbda == l)
#                 & (np.isclose(test_stat_mixture.rho, c))
#             ].t_p.values[0]
#             < 0.05
#         ):
#             axs[i // len(ks), i % len(ks)].set_title(
#                 f"k={k}, lmbda={l}, t-test significant"
#             )
#         else:
#             axs[i // len(ks), i % len(ks)].set_title(
#                 f"k={k}, lmbda={l}, t-test not significant"
#             )

#     plt.tight_layout()
#     plt.savefig(f"./doc/figures/density_comp_mixture_rho={round(c, 1)}.png")


# ######## Example ########
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# sns.violinplot(
#     x="lmbda",
#     y="cr_q",
#     hue="k",
#     data=cr_true_gen,
#     # inner="stick",
#     linewidth=0.5,
#     scale="width",
#     ax=axs[0],
# )
# axs[0].set_title("Copying rates for $q=p$")
# axs[0].set_xlabel("$\\lambda$")
# axs[0].set_ylabel("$cr_q$")

# sns.violinplot(
#     x="lmbda",
#     y="cr_q",
#     hue="k",
#     data=cr_mixture[cr_mixture.rho == 0.1],
#     # inner="stick",
#     linewidth=0.5,
#     scale="width",
#     ax=axs[1],
# )
# axs[1].set_title("Copying rates for $q = 0.1 * q_{copying} + 0.9 * p$")
# axs[1].set_xlabel("$\\lambda$")
# axs[1].set_ylabel("$cr_q$")

# # plot densities for lambda=13 and k=10
# sns.histplot(
#     cr_true_gen[(cr_true_gen.lmbda == 13) & (cr_true_gen.k == 10)].cr_q,
#     ax=axs[2],
#     kde=True,
#     label="p",
# )
# sns.histplot(
#     cr_mixture[
#         (cr_mixture.lmbda == 13) & (cr_mixture.k == 10) & (cr_mixture.rho == 0.1)
#     ].cr_q,
#     ax=axs[2],
#     kde=True,
#     label="$0.1 * q_{copying} + 0.9 * p$",
# )
# p_value = test_stat_mixture[
#     (test_stat_mixture.k == 10)
#     & (test_stat_mixture.lmbda == 13)
#     & (test_stat_mixture.rho == 0.1)
# ].t_p.values[0]
# axs[2].set_title(f"k={10}, $\\lambda$={13}, T-test p-value={round(p_value,2)}")
# axs[2].set_xlabel("$cr_q$")
# axs[2].legend()

# plt.tight_layout()
# plt.savefig("./doc/figures/cr_example.png")
