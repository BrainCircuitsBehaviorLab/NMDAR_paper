import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def fig_size(n_cols=1, ratio=None):
    """
    Get figure size for A4 page with n_cols columns and specified ratio (width/height).
    :param n_cols: Number of columns (0 for full page)
    :param ratio: Width/height ratio (None for default)
    :return:
    """

    if ratio is None:
        default_figsize = np.array(plt.rcParams['figure.figsize'])
        ratio = default_figsize[0] / default_figsize[1]  # 4:3

    mm_per_inch = 25.4

    # All measurements are in inches
    # A4_size = np.array((8.27, 11.69))  # A4 measurements
    A4_size = np.array((210, 297))  # A4 measurements

    # margins = 2  # On both dimensions
    margins = 50.8  # 2 inches on each dimension

    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width, height = size

    # Full page
    if n_cols == 0:
        figsize = np.array((width, height))
        if ratio == 1:  # Square
            figsize = np.array((size[0], size[0]))
        # return figsize

    # Full page / N columns width
    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = np.array((fig_width, fig_height))
        # return figsize

    return tuple(figsize / mm_per_inch)
