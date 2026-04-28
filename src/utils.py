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
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio  # 4:3

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 2  # On both dimension
    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width = size[0]
    height = size[1]

    # Full page (minus margins)
    if n_cols == 0:
        # Full A4 minus margins
        figsize = (width, height)
        if ratio == 1:  # Square
            figsize = (size[0], size[0])
        return figsize

    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = (fig_width, fig_height)
        return figsize
