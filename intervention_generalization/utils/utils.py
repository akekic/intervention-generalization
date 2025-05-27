import matplotlib as mpl
import numpy as np


def calc_stats(arr):
    mean = np.mean(arr)
    sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, sem


mm = 1 / (10 * 2.54)  # millimeters in inches
SINGLE_COLUMN = 85 * mm
DOUBLE_COLUMN = 6.75  # inches
FONTSIZE = 8
color_list = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
style_modifications = {
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "font.family": "sans-serif",  # set the font globally
    "font.sans-serif": "Helvetica",  # set the font name for a font family
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "errorbar.capsize": 2.5,
    "axes.prop_cycle": mpl.cycler(color=color_list),
}
