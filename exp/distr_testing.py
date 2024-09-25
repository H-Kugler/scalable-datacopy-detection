import pandas as pd
import itertools
from scipy.stats import shapiro, ttest_ind

# We have distributions of data-copying rates in `doc/exp`.
#
# We want to:
#
# 1. Test whether these distributions are from a normal distribution.
# 2. Test whether the samples drawn are from the same distribution,
#    i.e. differ the samples from the true generating model significantly from those from a copying generative model.
#

### load data
# df = pd.read_csv("../doc/exp/halfmoons_exps.csv")
df_true_gen = pd.read_csv("./doc/exp/exp_Halfmoons_Halfmoons(noise=0.1).csv")
df_mixed = pd.read_csv("./doc/exp/exp_MixedModel_Halfmoons(noise=0.1).csv")
df_mixture = pd.read_csv("./doc/exp/exp_Mixture_Halfmoons(noise=0.1).csv")
# q = p
cr_true_gen = df_true_gen.drop(columns=["C_T", "num_regions"])
cr_true_gen = (
    cr_true_gen.groupby(["noise", "trial", "lmbda", "k", "gamma"]).mean().reset_index()
)
# q = rho * q_copying + (1 - rho) * q_underfit
cr_mixed = df_mixed.drop(columns=["C_T", "num_regions"])
cr_mixed = (
    cr_mixed.groupby(["trial", "rho", "lmbda", "k", "gamma"]).mean().reset_index()
)
# q = rho * q_copying + (1 - rho) * p
cr_mixture = df_mixture.drop(columns=["C_T", "num_regions"])
cr_mixture = (
    cr_mixture.groupby(["trial", "rho", "lmbda", "k", "gamma"]).mean().reset_index()
)

# mixed model, i.e. q = rho * q_copying + (1 - rho) * q_underfit
lmbdas = cr_true_gen.lmbda.unique()
ks = cr_true_gen.k.unique()
rhos = cr_mixed.rho.unique()

test_stat = []

for k, l in itertools.product(ks, lmbdas):
    cr_true = cr_true_gen[(cr_true_gen.k == k) & (cr_true_gen.lmbda == l)].cr_q
    stat_true, p_true = shapiro(cr_true)
    for r in rhos:
        crs = cr_mixed[
            (cr_mixed.k == k) & (cr_mixed.lmbda == l) & (cr_mixed.rho == r)
        ].cr_q
        stat, p = ttest_ind(cr_true, crs)
        stat_mixed, p_mixed = shapiro(crs)
        test_stat.append(
            {
                "k": k,
                "lmbda": l,
                "rho": r,
                "t_stat": stat,
                "t_p": p,
                "shapiro_true_stat": stat_true,
                "shapiro_true_p": p_true,
                "shapiro_mixed_stat": stat_mixed,
                "shapiro_mixed_p": p_mixed,
            }
        )

test_stat = pd.DataFrame(test_stat)
test_stat.to_csv("./doc/exp/test_statistics_mixed_model.csv", index=False)

# mixture of models, i.e. q = rho * q_copying + (1 - rho) * p, with p being true generating distribution
lmbdas = cr_true_gen.lmbda.unique()
ks = cr_true_gen.k.unique()
rhos = cr_mixture.rho.unique()

test_stat = []

for k, l in itertools.product(ks, lmbdas):
    cr_true = cr_true_gen[(cr_true_gen.k == k) & (cr_true_gen.lmbda == l)].cr_q
    stat_true, p_true = shapiro(cr_true)
    for r in rhos:
        crs = cr_mixture[
            (cr_mixture.k == k) & (cr_mixture.lmbda == l) & (cr_mixture.rho == r)
        ].cr_q
        stat, p = ttest_ind(cr_true, crs)
        stat_mixed, p_mixed = shapiro(crs)
        test_stat.append(
            {
                "k": k,
                "lmbda": l,
                "rho": r,
                "t_stat": stat,
                "t_p": p,
                "shapiro_true_stat": stat_true,
                "shapiro_true_p": p_true,
                "shapiro_mixed_stat": stat_mixed,
                "shapiro_mixed_p": p_mixed,
            }
        )

test_stat = pd.DataFrame(test_stat)
test_stat.to_csv("./doc/exp/test_statistics_mixture.csv", index=False)
