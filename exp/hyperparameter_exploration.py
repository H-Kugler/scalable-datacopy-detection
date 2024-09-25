import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load csv file
df = pd.read_csv("./doc/exp/hyperparameter.csv")
df["diff"] = (df["rho"] - df["cr"]).abs()

sns.catplot(
    data=df,
    x="lmbda",
    y="diff",
    hue="n_proj",
    kind="bar",
    col="rho",
    col_wrap=3,
    errorbar="sd",
    palette="rocket",
    alpha=0.8,
    height=6,
)

plt.savefig("./doc/figures/hyperparameter.png")
