import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


def set_plotting_params():
    # Set LaTeX for text rendering and configure Palatino fonts
    rc("text", usetex=True)  # Enable LaTeX rendering
    rc("font", family="serif")  # Use serif fonts
    rc("font", serif="Palatino")  # Set Palatino as the serif font

    # Add dsfont package to the LaTeX preamble for double-struck symbols
    plt.rcParams["text.latex.preamble"] = r"\usepackage{dsfont}"

    # Set plotting style and font sizes
    plt.rcParams.update(
        {
            "font.size": 7,  # Base font size
            "axes.titlesize": 7,  # Title size
            "axes.labelsize": 7,  # Axis labels size
            "xtick.labelsize": 6,  # X-axis tick labels size
            "ytick.labelsize": 6,  # Y-axis tick labels size
            "legend.fontsize": 6,  # Legend font size
        }
    )

    # Calculate textwidth in inches for plot sizing
    textwidth_in_inches = 398.33864 / 72.27

    return textwidth_in_inches
