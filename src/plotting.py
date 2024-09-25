import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_C_T_results(sigmas: np.array, C_T: np.ndarray, ax=None, title=None):
    """
    Plots the results of the C_T statistics

    Parameters
    ----------
    C_T : np.ndarray
        C_T statistics
    ax : matplotlib.axes.Axes, optional
        Axes where the plot will be drawn, by default None
    title : str, optional
        Title of the plot, by default None

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    C_T_mean = np.mean(C_T, axis=1)
    C_T_std = np.std(C_T, axis=1)

    ax.plot(sigmas, C_T_mean, label="C_T")
    ax.fill_between(
        sigmas, C_T_mean - C_T_std, C_T_mean + C_T_std, alpha=0.2, label="std"
    )
    ax.set_xlabel("$\sigma$")
    ax.set_ylabel("C_T")
    ax.legend()
    ax.set_xscale("log")

    return ax


def plot_2D_data(X, y, ax=None, title=None):
    """
    Plots a 2D dataset

    Parameters
    ----------
    X : np.ndarray
        Data to be plotted
    y : np.ndarray
        Labels of the data
    ax : matplotlib.axes.Axes, optional
        Axes where the plot will be drawn, by default None
    title : str, optional
        Title of the plot, by default None

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if title is not None:
        ax.set_title(title)

    return ax
