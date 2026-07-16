import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence

import mplscience
import seaborn as sns
from anndata import AnnData

import regvelo as rgv

from ._utils import SIGNIFICANCE_PALETTE

def plot_visits_dist(
    df: pd.DataFrame,
    palette_rel: list[str],
    tick_range: float,
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot a boxplot of visit difference values per terminal state for a single TF.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns 'Value' and 'Group'.
    palette_rel : list of str
        Per-group hex colours derived from significance testing.
    tick_range : float
        Half-width of the x-axis range, centred at 0.5.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. If ``None`` (default), the height
        adapts to the number of terminal states shown (one box per state), clamped to
        a sensible range; pass a tuple to set it manually.
    """
    if figsize is None:
        # One horizontal box per terminal state: scale height with the number of states.
        n_groups = df["Group"].nunique()
        figsize = (3, float(np.clip(0.9 * n_groups + 0.6, 2, 12)))

    with mplscience.style_context():
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=df,
            y="Group",
            x="Value",
            palette=palette_rel,
            ax=ax,
            flierprops={
                "marker": ".",
                "markersize": 5,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },
        )
        ax.set_xlabel("Density change likelihood")
        ax.set_ylabel("Terminal state")

        xmin, xmax = 0.5 - tick_range, 0.5 + tick_range
        ticks = np.arange(xmin, xmax + 1e-6, tick_range)
        if 0.5 not in ticks:
            ticks = np.sort(np.append(ticks, 0.5))

        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)

        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.show()
